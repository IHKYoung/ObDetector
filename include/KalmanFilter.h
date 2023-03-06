//
// Created by young on 23-2-24.
//

#ifndef OBJECT_DETECTION_KALMANFILTER_H
#define OBJECT_DETECTION_KALMANFILTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

class myKalmanFilter {
public:
    myKalmanFilter() {}

    ~myKalmanFilter() {}

    void init(cv::Mat &x, cv::Mat &p, cv::Mat &f, cv::Mat &h, cv::Mat &q, cv::Mat &r) {
        x_ = x;
        p_ = p;
        f_ = f;
        h_ = h;
        q_ = q;
        r_ = r;
    }

    void predict() {
        x_ = f_ * x_;
        p_ = f_ * p_ * f_.t() + q_;
    }

    void update(cv::Mat &z) {
        cv::Mat y = z - h_ * x_;
        cv::Mat s = h_ * p_ * h_.t() + r_;
        cv::Mat k = p_ * h_.t() * s.inv();
        x_ = x_ + k * y;
        p_ = (cv::Mat::eye(p_.rows, p_.cols, CV_64F) - k * h_) * p_;
    }

    cv::Mat getState() {
        return x_;
    }

private:
    cv::Mat x_; // 状态向量
    cv::Mat p_; // 状态协方差矩阵
    cv::Mat f_; // 状态转移矩阵
    cv::Mat h_; // 观测矩阵
    cv::Mat q_; // 状态转移协方差矩阵
    cv::Mat r_; // 观测协方差矩阵
};

#endif //OBJECT_DETECTION_KALMANFILTER_H
