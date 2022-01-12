#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "VirtualSensor.h"
#include <vector>


using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

struct Vertex {
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector4f pos;
};


int main()
{
//    printCudaDeviceInfo(0);
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

//    int sensorFrame = 0;
    while(sensor.processNextFrame()) {
        float* depthMat = sensor.getDepth();
        unsigned int depthWidth = sensor.getDepthImageWidth();
        unsigned int depthHeight = sensor.getDepthImageHeight();

        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();     // get K matrix (intrinsics), global to camera frame
        Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();
        cout << depthIntrinsicsInv.size() << endl;
        break;

        cv::Mat depth_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, depthMat);
        cv::Mat filt_depth_mat(depth_mat.size(), CV_32F);
        cv::Mat depth_normals(depth_mat.size(), CV_32F);
        cv::Mat vertex_validity(depth_mat.size(), CV_32F);  //binary matrix for valid depth vals


        //appy bilateralFilter to raw depth
        cv::bilateralFilter(depth_mat, filt_depth_mat, 5, 50,50, BORDER_DEFAULT); //max d val of 5 recommended for real-time applications (9 for offline)
        cout << "depth mat size: " << depth_mat.size() << endl;
        cout << "filt_depthmat size: " << filt_depth_mat.size() << endl;

        //get vertices map in sensor/camera frame
        Vertex* vertices = new Vertex[depthWidth*depthHeight];

        unsigned int vertex_idx = -1;
        for (unsigned int r = 0; r < depthHeight; r++) {
            for (unsigned int c = 0; c < depthWidth; c++) {
                vertex_idx += 1;
                float depth_pixel = filt_depth_mat.at<float>(r,c);
                if (isnan(depth_pixel)) {
                    vertex_validity.at<int>(r,c) = 0;
//                    vertices[vertex_idx].pos = Vector4f(MINF, MINF, MINF, MINF);
                } else{
                    cout << depthIntrinsicsInv.size() << endl;
                    Vector3f camera_coord = depthIntrinsicsInv * Vector3f(r,c,1) * depth_pixel;
                    vertices[vertex_idx].pos[0] = camera_coord[0];
                    vertices[vertex_idx].pos[1] = camera_coord[1];
                    vertices[vertex_idx].pos[2] = camera_coord[2];
                    vertices[vertex_idx].pos[3] = 1.0f;
                    vertex_validity.at<int>(r,c) = 1;
                }
            }
        }

//        //compute normals of the vertex map (in camera frame)
//        vertex_idx = -1;
//        for (int x =0; x < depthHeight; ++x) {
//            for (int y = 0; y < depthWidth; ++y) {
//                vertex_idx += 1;
//                float dzdx = (vertices[vertex_idx].pos(x+1) - vertices[vertex_idx].pos(7)
//                float dzdx = (depth_mat.at<float>(x+1,y) - depth_mat.at<float>(x-1,y)) / 2.0f;
//                float dzdy = (depth_mat.at<float>(x,y+1) - depth_mat.at<float>(x,y-1)) / 2.0f;
//                Vec3f d(-dzdx, -dzdy, 1.0f);
//                Vec3f n = cv::normalize(d);
//                depth_normals.at<Vec3f>(x,y) = n;
//            }
//        }

        break;
        delete[] vertices;  //delete heap
    }

    return 0;
}