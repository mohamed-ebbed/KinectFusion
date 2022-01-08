#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

#include "VirtualSensor.h"
#include <vector>

int main()
{
    // Make sure this path points to the data folder
    std::string filenameIn = "../Data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameBaseOut = "mesh_";

    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(filenameIn)){
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    } else {
        std::cout << "File Opened" << std::endl;
    }

    sensor.processNextFrame();

    float* depthMat = sensor.getDepth();
    uint32_t depthWidth = sensor.getDepthImageWidth();
    uint32_t depthHeight = sensor.getDepthImageHeight();

    cv::Mat depth_Mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_8U, depthMat);
    cv::Mat filt_dept_Mat;

    cuda::GpuMat gpuImage;
    gpuImage.upload(depth_Mat);
    if (gpuImage.empty()) {
        cout << "could not upload data to gpu " << endl;
    } else {
        cout << "gpu works yayy!" << endl;
        //max d val of 5 recommended for real-time applications (9 for offline) for cpu.
        //for gpu can be higher
        cv::cuda::bilateralFilter(gpuImage, gpuImage, 20, 50, 50, BORDER_DEFAULT);
    }
    gpuImage.download(depth_Mat);
    cout << depth_Mat.size() << endl;

    cv::Mat depth_normals(depth_Mat.size(), CV_32FC3);

    //compute normals
    cout << "before normals" << endl;
    for (int x =0; x < depth_Mat.rows; ++x) {
        for (int y = 0; y < depth_Mat.cols; ++y) {
            float dzdx = (depth_Mat.at<float>(x+1,y) - depth_Mat.at<float>(x-1,y)) / 2.0f;
            float dzdy = (depth_Mat.at<float>(x,y+1) - depth_Mat.at<float>(x,y-1)) / 2.0f;
            Vec3f d(-dzdx, -dzdy, 1.0f);
            Vec3f n = cv::normalize(d);
            depth_normals.at<Vec3f>(x,y) = n;
        }
    }
    cout << "Normals ready?!" << endl;
    cout << "Normals size: " << depth_normals.size() << endl;



//    int sensorFrame {0};
//    while (sensor.processNextFrame()) {        { 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255 },
//
//        std::cout << "reading data.." << std::endl;
//        uint32_t depthWidth = sensor.getDepthImageWidth();
//        uint32_t depthHeight = sensor.getDepthImageHeight();
//
//        cout << "size of depthMap: " << sizeof(depthMat) / sizeof(float) << endl;
//        cout << "width: " << depthWidth << endl;
//        cout << "height: " << depthHeight << endl;
//        cout << "--------------------" << endl;
//    }




//    printCudaDeviceInfo(0);
    return 0;
}
