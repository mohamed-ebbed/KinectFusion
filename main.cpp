#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include "VirtualSensor.h"
//#include "PoseEstimation.h"
#include "core/VolumetricFusion.h"
#include "core/declarations.h"
#include <vector>
#include "core/RaycastingSimple.h"

using namespace std;
using namespace cv;
using namespace cuda;


int main()
{

        // ----------------   volumetric fusion -------------------
    int grid_size = 256;
    float min_x = -2;
    float max_x = 2;
    float min_y = -2;
    float max_y = 2;
    float min_z = -2;
    float max_z = 2;

    float trunc_val = 3.0f;

    float minDepth = 0.1f;
    float maxDepth = 1.5;


    VolumetricFusion* volFusion = new VolumetricFusion(grid_size, min_x, max_x, min_y, max_y,min_z, max_z,
                                   trunc_val);

    Raycasting* raycasting = new Raycasting(minDepth, maxDepth, min_x, max_x, min_y, max_y, min_z, max_z, trunc_val, grid_size);

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


    int sensor_frame = 0;

    Matrix4f initialPose;
    while(sensor.processNextFrame()) {
        sensor_frame+= 1;
        float* depthMat = sensor.getDepth();
        unsigned int depthWidth = sensor.getDepthImageWidth();
        unsigned int depthHeight = sensor.getDepthImageHeight();

        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();     // get K matrix (intrinsics), global to camera frame
        Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();

        Matrix4f depthExtrinsics = sensor.getTrajectory();

        if(sensor_frame == 1)
            initialPose = depthExtrinsics;
        cout << depthExtrinsics << endl;
        // Matrix4f depthExtrinsicsInv = depthExtrinsics.inverse();

        cv::Mat depth_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, depthMat);
        cv::Mat filt_depth_mat(depth_mat.size(), CV_32F);
        cv::Mat depth_normals(depth_mat.size(), CV_32FC3);
        // cv::Mat vertex_validity(depth_mat.size(), CV_32F);  //binary matrix for valid depth vals
        int vertex_validity[depthHeight*depthWidth];

        //appy bilateralFilter to raw depth
        cv::bilateralFilter(depth_mat, filt_depth_mat, 9, 75, 75, BORDER_DEFAULT); //max d val of 5 recommended for real-time applications (9 for offline)
        cout << "depth mat size: " << depth_mat.size() << endl;
        cout << "filt_depthmat size: " << filt_depth_mat.size() << endl;

        //get vertices map in sensor/camera frame
        Vertex* vertices = new Vertex[depthWidth*depthHeight];
        Normal* normals = new Normal[depthWidth*depthHeight];

        unsigned int vertex_idx = -1;
        for (unsigned int r = 0; r < depthHeight; r++) {
            for (unsigned int c = 0; c < depthWidth; c++) {
                vertex_idx += 1;
                float depth_pixel = filt_depth_mat.at<float>(r, c);
                normals[vertex_idx].val = Vector3f(0.0f, 0.0f, 0.0f);

                if (isnan(depth_pixel)) {
                    // vertex_validity.at<int>(r, c) = 0;
                    vertex_validity[vertex_idx] = 0;
                    //vertices[vertex_idx].pos = Vector4f(MINF, MINF, MINF, MINF);
                } else {
                    Vector3f camera_coord = depthIntrinsicsInv * Vector3f(c, r, 1) * depth_pixel;
                    vertices[vertex_idx].pos[0] = camera_coord[0];
                    vertices[vertex_idx].pos[1] = camera_coord[1];
                    vertices[vertex_idx].pos[2] = camera_coord[2];
                    vertices[vertex_idx].pos[3] = 1.0f;
                    // vertex_validity.at<int>(r, c) = 1;
                    vertex_validity[vertex_idx] = 1;
                }
            }
        }

        // -------------- compute normals ------------
        //         cv::Mat normalsMap_Vis = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32FC3, cv::Scalar(1,1,1));
        unsigned int numVertices = depthWidth*depthHeight;
        float cpp_normals[numVertices][3];


        Vector3f* predictedNormals = new Vector3f[depthWidth*depthHeight];
        Vector3f* predictedVertices = new Vector3f[depthWidth*depthHeight];
        float* phongSurface = new float[depthWidth*depthHeight];


        unsigned int curr_idx = -1;
         for(unsigned int v = 1; v < depthHeight-1; v++) {
             for (unsigned int u = 1; u < depthWidth-1 ; u++) {
                curr_idx = u + v*depthWidth;
                Vector3f v1, v2;
                Vector3f vk = Vector3f(vertices[curr_idx].pos[0], vertices[curr_idx].pos[1], vertices[curr_idx].pos[2]);
                Vector3f vk_right = Vector3f(vertices[curr_idx+1].pos[0], vertices[curr_idx+1].pos[1], vertices[curr_idx+1].pos[2]);
                Vector3f vk_up = Vector3f(vertices[curr_idx+depthWidth].pos[0], vertices[curr_idx+depthWidth].pos[1], vertices[curr_idx+depthWidth].pos[2]);

                v1 = vk_right - vk;
                v2 = vk_up - vk;

                normals[curr_idx].val = v1.cross(v2);
                normals[curr_idx].val.normalize();
             }
         }




        volFusion->step(depthExtrinsics, depthMat, normals, vertex_validity, depthWidth, depthHeight, depthIntrinsics);

        cout << "Finished volumetric fusion step" << endl;

        raycasting->ProcessSDF(volFusion->getF(), initialPose, depthIntrinsics, predictedVertices, predictedNormals,depthWidth, depthHeight, phongSurface);

        cout << "Finished raycasting fusion step" << endl;

        for (unsigned int i = 0; i < numVertices; i++) {

            cpp_normals[i][0] = predictedNormals[i][0];
            cpp_normals[i][1] = predictedNormals[i][1];
            cpp_normals[i][2] = predictedNormals[i][2];
        }

        cv::Mat normalsMap_Vis = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32FC3, cpp_normals);

        cv::Mat phong_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, phongSurface);

        if(sensor_frame % 10 == 0){

            cv::imshow("PredictedNormals Map", normalsMap_Vis);
            cv::imshow("Phong Surface", phong_mat);


            waitKey(0);
        }


        // delete[] vertices;
        // delete[] normals;
        // //delete volFusion;
    }

    return 0;
}