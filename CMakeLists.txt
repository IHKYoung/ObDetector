cmake_minimum_required(VERSION 3.10)
project(object_detection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")

find_package(OpenCV 4.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set(Inudev_Include /opt/Inuitive/InuDev/include/)
include_directories(
        ${Inudev_Include}
        include
)

link_directories(${PROJECT_SOURCE_DIR})
add_executable(getCamera src/getCamera.cpp)
target_link_libraries(getCamera
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        /opt/Inuitive/InuDev/lib/libCommonUtilities.so
        /opt/Inuitive/InuDev/lib/libInuStreams.so
        )

add_executable(objDetect src/objDetect.cpp)
target_link_libraries(objDetect
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        /opt/Inuitive/InuDev/lib/libCommonUtilities.so
        /opt/Inuitive/InuDev/lib/libInuStreams.so
        )

add_executable(objDetectSplit src/objDetectSplit.cpp)
target_link_libraries(objDetectSplit
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        /opt/Inuitive/InuDev/lib/libCommonUtilities.so
        /opt/Inuitive/InuDev/lib/libInuStreams.so
        )

add_executable(objDetectKalman src/objDetectKalman.cpp)
target_link_libraries(objDetectKalman
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        /opt/Inuitive/InuDev/lib/libCommonUtilities.so
        /opt/Inuitive/InuDev/lib/libInuStreams.so
        )