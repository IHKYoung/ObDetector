# ObDetector
> :bulb:Algorithms for **Ob**stacle/**Ob**ject detection **using traditional point cloud processing methods.**


The current code version has completed the basic algorithm implementation and can be used as a baseline
The method used:
1. Filter acceleration
2. Clustering
3. Detection
4. Tracking(Using Kalman Filters and Algorithmic Tricks)

## Instructions:
Refer to the CMake project to compile, you need to use PCL and OpenCV.

## :memo:Need to improve:
1. Remove the test code and sort out the stripped-down version
2. For the problem of point cloud jitter, there is no suitable method to solve it (mainly reflected in tracking)
3. Improve the algorithm to the ROS version and remove the dependency on the camera SDK

