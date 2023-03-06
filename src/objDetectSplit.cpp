#include "InuStreams.h"
#include "InuSensor.h"
#include "InuSensorExt.h"
#include "DepthStreamExt.h"
#include "DepthStream.h"
#include "Version.h"

// PCL
#include <pcl/point_types.h>
#include <pcl/common/impl/common.hpp>
#include <pcl/common/transforms.h> // pcl::transformPointCloud
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include <iostream>
#include <string>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "DBSCAN_basic.h"
#include "DBSCAN_precomp.h"
#include "DBSCAN_kdtree.h"
#include "TicToc.h"

using namespace InuDev;
using namespace std;
using namespace cv;

std::shared_ptr<CInuSensorExt> mSensor;
std::shared_ptr<CDepthStreamExt> depthStream;
std::shared_ptr<CImageStream> webcamStream;
CCalibrationData mCalibrationData;
CHwInformation mHwInfo;
std::shared_ptr<cv::Mat> ptr_image;
pcl::visualization::PCLVisualizer *obstacle_viewer = nullptr;
std::mutex image_mutex;
std::condition_variable cv_image;
std::mutex viewer_mutex;
bool get_image = false;

struct MinMaxPoints {
    pcl::PointXYZ min_point;  // 最小空间点
    pcl::PointXYZ max_point;  // 最大空间点
    float offset_threshold{};  // 偏移阈值
};
std::vector<MinMaxPoints> prev_bboxes;  // 保存之前的最小最大点

bool isSameBoundingBox(const MinMaxPoints &prev_bbox, const MinMaxPoints &curr_bbox) {
    float offset_threshold = prev_bbox.offset_threshold;
    float x_offset = std::abs(prev_bbox.min_point.x - curr_bbox.min_point.x);
    float y_offset = std::abs(prev_bbox.min_point.y - curr_bbox.min_point.y);
    float z_offset = std::abs(prev_bbox.min_point.z - curr_bbox.min_point.z);
    if (x_offset > offset_threshold || y_offset > offset_threshold || z_offset > offset_threshold)
        return false;
    x_offset = std::abs(prev_bbox.max_point.x - curr_bbox.max_point.x);
    y_offset = std::abs(prev_bbox.max_point.y - curr_bbox.max_point.y);
    z_offset = std::abs(prev_bbox.max_point.z - curr_bbox.max_point.z);
    if (x_offset > offset_threshold || y_offset > offset_threshold || z_offset > offset_threshold)
        return false;
    return true;
}

bool isSameBoundingBoxSimple(MinMaxPoints prev_bbox, MinMaxPoints curr_bbox) {
    // 判断最小最大点的偏移是否小于阈值
    float x_offset = fabs(prev_bbox.min_point.x - curr_bbox.min_point.x) +
                     fabs(prev_bbox.max_point.x - curr_bbox.max_point.x);
    float y_offset = fabs(prev_bbox.min_point.y - curr_bbox.min_point.y) +
                     fabs(prev_bbox.max_point.y - curr_bbox.max_point.y);
    float z_offset = fabs(prev_bbox.min_point.z - curr_bbox.min_point.z) +
                     fabs(prev_bbox.max_point.z - curr_bbox.max_point.z);

    return (x_offset < prev_bbox.offset_threshold &&
            y_offset < prev_bbox.offset_threshold &&
            z_offset < prev_bbox.offset_threshold);
}

bool isSameBoundingBoxComplex(const MinMaxPoints &prev_points, const MinMaxPoints &curr_points) {
    // 计算 prev 和 curr 两个 bounding box 中心点的距离
    float prev_center_x = (prev_points.min_point.x + prev_points.max_point.x) / 2;
    float prev_center_y = (prev_points.min_point.y + prev_points.max_point.y) / 2;
    float prev_center_z = (prev_points.min_point.z + prev_points.max_point.z) / 2;

    float curr_center_x = (curr_points.min_point.x + curr_points.max_point.x) / 2;
    float curr_center_y = (curr_points.min_point.y + curr_points.max_point.y) / 2;
    float curr_center_z = (curr_points.min_point.z + curr_points.max_point.z) / 2;

    float center_distance = std::sqrt(
            std::pow(curr_center_x - prev_center_x, 2) + std::pow(curr_center_y - prev_center_y, 2) +
            std::pow(curr_center_z - prev_center_z, 2));

    // 如果距离小于偏移阈值，则认为是同一个 bounding box
    if (center_distance < curr_points.offset_threshold) {
        // 计算 prev 和 curr 两个 bounding box 的大小
        float prev_box_size = (prev_points.max_point.x - prev_points.min_point.x) *
                              (prev_points.max_point.y - prev_points.min_point.y) *
                              (prev_points.max_point.z - prev_points.min_point.z);
        float curr_box_size = (curr_points.max_point.x - curr_points.min_point.x) *
                              (curr_points.max_point.y - curr_points.min_point.y) *
                              (curr_points.max_point.z - curr_points.min_point.z);

        // 如果大小相差不大，则认为是同一个 bounding box
        float size_diff = std::abs(curr_box_size - prev_box_size) / prev_box_size;
        if (size_diff < 0.2) { // 20% 的大小误差阈值
            return true;
        }
    }
    return false;
}

void pclViewerThread() {
    obstacle_viewer = new pcl::visualization::PCLVisualizer("Obstacle Viewer");
    while (!obstacle_viewer->wasStopped()) {
        std::unique_lock<std::mutex> pcl_lock(viewer_mutex);
        obstacle_viewer->spinOnce();
    }
    delete obstacle_viewer;
}

void DepthCallback(const std::shared_ptr<CDepthStream> &, const std::shared_ptr<const CImageFrame> &dFrame,
                   const CInuError &iError) {
    if (iError != eOK || !dFrame->Valid)
        return;
    cv::Mat depth((int) dFrame->Height(), (int) dFrame->Width(), CV_16UC1, (unsigned char *) dFrame->GetData());
    cv::Mat depth_display;

    depth.convertTo(depth_display, CV_8UC1, 8.0 / 256.0, 0);
    // depth.convertTo(depth_display, CV_8UC4, 255.0 / 6000, 0); // up to 5m

    std::unique_lock<std::mutex> rgb_lock(image_mutex);
    cv_image.wait(rgb_lock, [] { return get_image; });
    cv::Mat rgb;
    cv::resize(*ptr_image, rgb, Size(depth.cols, depth.rows));

    // TODO Depth to PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_near(new pcl::PointCloud<pcl::PointXYZ>), cloud_far(
            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZRGB>);

    const double cx = mCalibrationData.Sensors[0].VirtualCamera.Intrinsic.OpticalCenter[0];
    const double cy = mCalibrationData.Sensors[0].VirtualCamera.Intrinsic.OpticalCenter[1];
    const double fx = mCalibrationData.Sensors[0].VirtualCamera.Intrinsic.FocalLength[0];
    const double fy = mCalibrationData.Sensors[0].VirtualCamera.Intrinsic.FocalLength[1];

    for (int y = 0; y < depth.rows; y++) {
        for (int x = 0; x < depth.cols; x++) {
            unsigned short z = depth.ptr<unsigned short>(y)[x]; // mm
            if (z <= 1000) {

                pcl::PointXYZ p;
                p.z = z / 1000.0;
                p.x = p.z * (x - cx) / fx;
                p.y = p.z * (y - cy) / fy;

                cloud_near->points.push_back(p);
            } else if (z <= 3000) {
                pcl::PointXYZ p;
                p.z = z / 1000.0;
                p.x = p.z * (x - cx) / fx;
                p.y = p.z * (y - cy) / fy;

                cloud_far->points.push_back(p);
            } else {
                continue;
            }
        }
    }
    *cloud = *cloud_near + *cloud_far;
    // std::cout << "Point Cloud before filtering has: " << cloud->points.size() << " data points." << std::endl;
    // std::cout << "Cloud before filtering has: " << cloud_near->points.size() << " data points." << std::endl;
    // std::cout << "Cloud Far before filtering has: " << cloud_far->points.size() << " data points." << std::endl;

    TicToc t_vg_near;
    pcl::VoxelGrid<pcl::PointXYZ> vg_near;
    vg_near.setInputCloud(cloud_near);
    vg_near.setLeafSize(0.01f, 0.01f, 0.01f); // unit: m
    vg_near.filter(*cloud_near);
    // std::cout << "Cloud Near after VoxelGrid has: " << cloud_near->points.size() << " data points." << std::endl;
    // std::cout << "VoxelGrid time cost:" << t_vg_near.toc() << " ms" << std::endl;

    TicToc t_vg_far;
    pcl::VoxelGrid<pcl::PointXYZ> vg_far;
    vg_far.setInputCloud(cloud_far);
    vg_far.setLeafSize(0.03f, 0.03f, 0.3f); // unit: m
    vg_far.filter(*cloud_far);
    // std::cout << "Cloud Far after VoxelGrid has: " << cloud_far->points.size() << " data points." << std::endl;
    // std::cout << "VoxelGrid time cost:" << t_vg_far.toc() << " ms" << std::endl;

    // 用于空间坐标系旋转的纠正
    // X-R Y-G Z-B
    Eigen::Affine3f rotation_correct = Eigen::Affine3f::Identity();
    // // 注意是旋转前的方向
    // rotation_correct.translation() << 0.0, 0.0, -2.0;
    rotation_correct.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
    // TODO 在过滤平面的时候存在问题，等实时的时候再进行调试
    pcl::transformPointCloud(*cloud, *cloud, rotation_correct);
    pcl::transformPointCloud(*cloud_near, *cloud_near, rotation_correct);
    pcl::transformPointCloud(*cloud_far, *cloud_far, rotation_correct);
    pcl::transformPointCloud(*cloud_filtered, *cloud_filtered, rotation_correct);

    // RANSAC
    TicToc t_ransac;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud_near);
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_near);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_near);
    std::cout << "SACSegmentation time cost:" << t_ransac.toc() << " ms" << std::endl;

    *cloud_filtered = *cloud_near + *cloud_far;

    // DBSCAN with Kdtree
    TicToc t_dbscan;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_near(new pcl::search::KdTree<pcl::PointXYZ>);
    tree_near->setInputCloud(cloud_near);
    std::vector<pcl::PointIndices> cluster_near_indices;
    DBSCANKdtreeCluster<pcl::PointXYZ> ec_near;
    ec_near.setCorePointMinPts(20);
    ec_near.setClusterTolerance(0.04);
    ec_near.setMinClusterSize(50);
    ec_near.setMaxClusterSize(10000);
    ec_near.setSearchMethod(tree_near);
    ec_near.setInputCloud(cloud_near);
    ec_near.extract(cluster_near_indices);

    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_far(new pcl::search::KdTree<pcl::PointXYZ>);
    // tree_far->setInputCloud(cloud_far);
    // std::vector<pcl::PointIndices> cluster_far_indices;
    // DBSCANKdtreeCluster<pcl::PointXYZ> ec_far;
    // ec_far.setCorePointMinPts(50);
    // ec_far.setClusterTolerance(0.2);
    // ec_far.setMinClusterSize(200);
    // ec_far.setMaxClusterSize(300000);
    // ec_far.setSearchMethod(tree_far);
    // ec_far.setInputCloud(cloud_far);
    // ec_far.extract(cluster_far_indices);

    std::vector<pcl::PointIndices> cluster_indices;
    cluster_indices.insert(cluster_indices.end(), cluster_near_indices.begin(), cluster_near_indices.end());
    // cluster_indices.insert(cluster_indices.end(), cluster_far_indices.begin(), cluster_far_indices.end());

    std::cout << "DBSCANKdtreeCluster time cost:" << t_dbscan.toc() << " ms" << std::endl;

    std::unique_lock<std::mutex> pcl_lock(viewer_mutex);
    if (!obstacle_viewer) {
        std::thread viewer_thread(pclViewerThread);
        viewer_thread.detach();
        return;
    }

    int j = 0;
    obstacle_viewer->removePointCloud("original cloud");
    obstacle_viewer->removeAllShapes();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ponit_color(cloud, 0, 0, 200);
    obstacle_viewer->addPointCloud<pcl::PointXYZ>(cloud, ponit_color, "original cloud");
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
         it != cluster_indices.end(); it++, j++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (int indice: it->indices) {
            pcl::PointXYZRGB tmp;
            tmp.x = cloud_filtered->points[indice].x;
            tmp.y = cloud_filtered->points[indice].y;
            tmp.z = cloud_filtered->points[indice].z;
            tmp.r = j * 100 % 255;
            tmp.g = j * 120 % 255;
            tmp.b = j * 150 % 255;
            cloud_clustered->points.push_back(tmp);
            cloud_cluster->points.push_back(cloud_filtered->points[indice]);
        }

        std::cout << "The Cluster of Obstacle has: " << cloud_cluster->points.size() << " data points."
                  << std::endl;

        pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
        feature_extractor.setInputCloud(cloud_cluster);
        feature_extractor.compute();
        pcl::PointXYZ min_point_AABB;
        pcl::PointXYZ max_point_AABB;
        pcl::PointXYZ min_point_OBB;
        pcl::PointXYZ max_point_OBB;
        pcl::PointXYZ position_OBB;
        Eigen::Matrix3f rotational_matrix_OBB;
        MinMaxPoints curr_bbox;  // 当前帧的最小最大点
        curr_bbox.offset_threshold = 0.2;  // 偏移阈值设为0.1
        feature_extractor.getAABB(min_point_AABB, max_point_AABB);
        // curr_bbox.min_point = min_point_AABB;
        // curr_bbox.max_point = max_point_AABB;

        feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
        curr_bbox.min_point = min_point_OBB;
        curr_bbox.max_point = max_point_OBB;

        // if (prev_bboxes.empty()) {  // 第一次计算，直接保存
        //     prev_bboxes.push_back(curr_bbox);
        //     return;
        // }
        // bool is_same_bbox = false;
        // for (const auto &prev_bboxe: prev_bboxes) {
        //     if (isSameBoundingBox(prev_bboxe, curr_bbox)) {
        //         // 与之前的最小最大点偏移小于阈值，认为是同一个bounding box
        //         is_same_bbox = true;
        //         curr_bbox = prev_bboxe;
        //         break;
        //     }
        // }
        // if (!is_same_bbox) {
        //     // 与之前的最小最大点偏移超过阈值，认为是新的bounding box
        //     prev_bboxes.push_back(curr_bbox);
        // }
        // /// TODO 定时清理
        // if (prev_bboxes.size() > 5) {
        //     // std::cout << "History Size: " << prev_bboxes.size() << std::endl;
        //     prev_bboxes.erase(prev_bboxes.begin(), prev_bboxes.begin() + (int) prev_bboxes.size() / 2);
        // }

        double min_u, min_v, max_u, max_v;
        min_u = (fx * min_point_AABB.x / min_point_AABB.z) + cx;
        min_v = (fy * min_point_AABB.y / min_point_AABB.z) + cy;
        max_u = (fx * max_point_AABB.x / max_point_AABB.z) + cx;
        max_v = (fy * max_point_AABB.y / max_point_AABB.z) + cy;
        cv::Point p_min(min_u, max_u);
        cv::Point p_max(min_v, max_v);
        cv::rectangle(depth_display, p_min, p_max, Scalar(255, 0, 0), 2, cv::LINE_8);
        std::string dis_text = "dis: " + std::to_string(abs(min_point_AABB.z + max_point_AABB.z)) + " m";
        cv::putText(depth_display, dis_text, cv::Point((min_u + min_v) / 2, (max_u + max_v) / 2),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1, false);

        // Use AABB
        // std::string aabb_id = "AABB-" + std::to_string(j);
        // std::cout << "AABB id = " << aabb_id << std::endl;
        // obstacle_viewer->addCube(curr_bbox.min_point.x, curr_bbox.max_point.x, curr_bbox.min_point.y,
        //                          curr_bbox.max_point.y,
        //                          curr_bbox.min_point.z, curr_bbox.max_point.z, 1.0, 1.0, 1.0, aabb_id);
        // obstacle_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, 0.5,
        //                                              aabb_id);
        // obstacle_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, aabb_id);
        // obstacle_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, aabb_id);

        // Use OBB
        Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
        Eigen::Quaternionf quat(rotational_matrix_OBB);
        std::string obb_id = "OBB-" + std::to_string(j);
        std::cout << "OBB id = " << obb_id << std::endl;
        obstacle_viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y,
                                 max_point_OBB.z - min_point_OBB.z, obb_id);
        obstacle_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, 0.5,
                                                     obb_id);
        obstacle_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, obb_id);
        obstacle_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, obb_id);


    }
    obstacle_viewer->spinOnce();

    applyColorMap(depth_display, depth_display, cv::COLORMAP_JET);
    cv::imshow("Depth", depth_display);

    cv::imshow("RGB ", rgb);
    get_image = false;

    cv::waitKey(1);
}

void WebCamCallback(const std::shared_ptr<CImageStream> &, const std::shared_ptr<const CImageFrame> &iFrame,
                    const CInuError &iError) {
    if (iError != eOK || !iFrame->Valid)
        return;

    CInuError err(eOK);
    cv::Mat img;
    img = cv::Mat(iFrame->Height(), iFrame->Width(), CV_8UC4,
                  (uchar *) iFrame->GetData()); //  CV_8UC1  CV_16UC1  CV_8UC4

    cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
    {
        std::lock_guard<std::mutex> rgb_lock(image_mutex);
        ptr_image = std::make_shared<cv::Mat>(img);
        get_image = true;
    }
    cv_image.notify_one();
}

void StreamCallback(const shared_ptr<CBaseStream> &pStream, const std::shared_ptr<const CBaseFrame> &pFrame,
                    const CInuError &error) {

    if (nullptr != dynamic_pointer_cast<CDepthStream>(pStream)) {
        DepthCallback(static_pointer_cast<CDepthStream>(pStream), static_pointer_cast<const CImageFrame>(pFrame),
                      error);
    } else if (nullptr != dynamic_pointer_cast<CImageStream>(pStream)) {
        WebCamCallback(static_pointer_cast<CImageStream>(pStream), static_pointer_cast<const CImageFrame>(pFrame),
                       error);
    }
}

void sensor_call(const std::shared_ptr<CInuSensor> &, EConnectionState eState, const CInuError &) {
    std::cout << eState << std::endl;
}

std::map<EChannelType, std::string> mChannelTypeName =
        {
                {eUnknownChannelType,      "eUnknownChannelType"},
                {eGeneralCameraChannel,    "eGeneralCameraChannel"},
                {eTrackingChannel,         "eTrackingChannel"},
                {eStereoChannel,           "eStereoChannel"},
                {eDepthChannel,            "eDepthChannel"},
                {eFeaturesTrackingChannel, "eFeaturesTrackingChannel"},
        };

/* init_sandbox */
bool init_sandbox() {
    std::map<CEntityVersion::EEntitiesID, CEntityVersion> oVersion;
    try {
        mSensor = CInuSensorExt::Create();
    }
    catch (std::exception const &e) {
    }

    CInuError err(eOK);

    std::cout << "Version: " IAF_VERSION_STR << std::endl;
    std::map<uint32_t, CHwChannel> oChannels;
    CDeviceParamsExt params;
    params.FPS = 30;
    params.SensorRes = eBinning;

    do {
        std::vector<CDpeParams> vecDpeParams;

        err = mSensor->Init(mHwInfo, vecDpeParams, params);

        if (err != eOK) {
            std::cout << "InuSensor failed to Init: " << string(err) << std::endl;
        } else {
            std::cout << "Print Active Channels" << std::endl;
            for (const auto &channel: mHwInfo.GetChannels()) {
                std::cout << "\t"
                          << "Channel ID " << channel.second.ChannelId << std::endl;
                std::cout << "\t"
                          << "Channel Type " << mChannelTypeName[channel.second.ChannelType] << std::endl;
                std::cout << "\t"
                          << "Channel Connected Sensors: " << channel.second.ChannelSensorsID.size() << std::endl;

                for (const auto &sensor: mHwInfo.GetSensorsPerChannel(channel.second.ChannelId)) {
                    std::cout << "\t\t"
                              << "Sensor ID  " << sensor.second.Id << std::endl;
                    std::cout << "\t\t"
                              << "Sensor LensType  " << sensor.second.LensType << std::endl;
                    std::cout << "\t\t"
                              << "Sensor Model  " << sensor.second.Model << std::endl;
                    std::cout << "\t\t"
                              << "Sensor Role  " << sensor.second.Role << std::endl;
                }
                std::cout << "\t"
                          << "Channel Connected Injectors: " << channel.second.ChannelInjectorsID.size() << std::endl;
                for (const auto &injector: mHwInfo.GetInjectorsPerChannel(channel.second.ChannelId)) {
                    std::cout << "\t\t"
                              << "Injector ID  " << injector.second.Id << std::endl;
                    std::cout << "\t\t"
                              << "Injector StreamerName  " << injector.second.StreamerName << std::endl;
                }
                std::cout << std::endl;
            }

            std::cout << "Print Default DPE params" << std::endl;
            int i = 0;
            for (const auto &dpe: vecDpeParams) {
                std::cout << "\t"
                          << "#  " << i++ << std::endl;
                std::cout << "\t"
                          << "File Name " << dpe.fileName << std::endl;
                std::cout << "\t"
                          << "Frame Numbers " << dpe.frameNum << std::endl;
            }
            std::cout << std::endl;

            mSensor->GetVersion(oVersion);
            std::cout << oVersion[CEntityVersion::eSerialNumber].Name << ": "
                      << oVersion[CEntityVersion::eSerialNumber].VersionName << std::endl;

            mSensor->GetCalibrationData(mCalibrationData);

            std::map<uint32_t, CChannelControlParams> channelParams;
            std::map<uint32_t, CChannelSize> channelsSize;

            CStartDeviceParamsExt startDeviceParams;
            if (!vecDpeParams.empty()) {
                startDeviceParams.VecDpeParams = vecDpeParams;
            }
            CChannelControlParams feCP;
            feCP.InterleaveMode = eInterleave;
            feCP.FPS = 30;
            feCP.SensorRes = eBinning;

            CChannelControlParams feye;
            feye.InterleaveMode = eInterleave;
            feye.FPS = 30;

            for (const auto &channel: mHwInfo.GetChannels()) {
                if (eStereoChannel == channel.second.ChannelType) {
                    startDeviceParams.ChannelControlParam[channel.first] = feCP;
                    // break;
                }
                if (2 == channel.second.ChannelType) {

                    startDeviceParams.ChannelControlParam[channel.first] = feye;
                }
            }

            err = mSensor->Start(channelsSize, startDeviceParams);

            if (err != eOK) {
                std::cout << "InuSensor failed to Start: " << string(err) << std::endl;
            } else {
                std::cout << "Print Channel Size" << std::endl;
                for (auto cs: channelsSize) {
                    std::cout << "\t"
                              << "Channel " << cs.first << std::endl;
                    std::cout << "\t"
                              << "Channel " << cs.second << std::endl;
                    std::cout << std::endl;
                }
            }
        }
    } while (err != eOK);

    std::cout << "InuSensor Init" << std::endl;

    mSensor->Register(sensor_call);

    std::cout << "Sensor Started" << std::endl;

    // Depth
    depthStream = mSensor->CreateDepthStreamExt();
    err = depthStream->Init();
    if (err != eOK) {
        return false;
    }
    CDisparityParams depth_params;
    depthStream->GetDisparityParams(depth_params);
    depth_params.MaxDistance = 10 * 1000;
    depthStream->SetDisparityParams(depth_params);

    err = depthStream->Start();
    if (err != eOK) {
        std::cout << "depth stream failed to Start" << std::endl;
        return false;
    }
    std::cout << "Depth Started" << std::endl;
    depthStream->Register(StreamCallback);

    // Webcam
    int imageChannel = 4;
    webcamStream = mSensor->CreateImageStream(imageChannel); // 7
    // err = webcamStream->Init(CImageStream::EPostProcessing::eRegistered, 5); // RGB和D对齐
    err = webcamStream->Init(CImageStream::eDefault);
    if (err != eOK) {
        std::cout << "webcam stream failed to Init" << std::endl;
        return false;
    }
    err = webcamStream->Start();
    if (err != eOK) {
        std::cout << "webcam stream failed to Start" << std::endl;
        return false;
    }
    std::cout << "WebCam Started" << std::endl;
    webcamStream->Register(StreamCallback);
    CSensorControlParams control;
    control.AutoControl = false;
    control.ExposureTime = 16500;
    mSensor->SetSensorControlParams(control, 3);

    return (true);
}

/* main */
int main(int argc, char *argv[]) {
    CInuError err(eOK);

    if (!init_sandbox()) {
        exit(-1);
    }
    char quit;
    while (quit != 'q') {
        cout << "\nPress 'q' to Exit...\n";
        cin >> quit;
    }

    return 0;
}
