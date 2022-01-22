#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "VirtualSensor.h"
//#include "PoseEstimation.h"
#include "core/VolumetricFusion.h"
#include "core/declarations.h"
#include <vector>

using namespace std;
using namespace cv;
using namespace dnn;


int main()
{
//    printCudaDeviceInfo(0);
    // Make sure this path points to the data folder
    std::string filenameIn = "../../Data/rgbd_dataset_freiburg1_xyz/";
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
    while(sensor.processNextFrame()) {
        float* depthMat = sensor.getDepth();
        unsigned int depthWidth = sensor.getDepthImageWidth();
        unsigned int depthHeight = sensor.getDepthImageHeight();

        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();     // get K matrix (intrinsics), global to camera frame
        Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();

        Matrix4f depthExtrinsics = sensor.getTrajectory();
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

        for (unsigned int i = 0; i < numVertices; i++) {
//            normals[i].val(0) = (normals[i].val(0) + 1.0f) / 2.0f;
//            normals[i].val(1) = (normals[i].val(1) + 1.0f) / 2.0f;
//            normals[i].val(2) = (normals[i].val(2) + 1.0f) / 2.0f;
            cpp_normals[i][0] = normals[i].val(0);
            cpp_normals[i][1] = normals[i].val(1);
            cpp_normals[i][2] = normals[i].val(2);
        }
        cv::Mat normalsMap_Vis = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32FC3, cpp_normals);

        // ----------------   volumetric fusion -------------------
        int grid_size = 256;
        float x_dist = 1;
        float y_dist = 1;
        float z_dist = 0.0f;
        float trunc_val = 5.0f;

        VolumetricFusion* volFusion = new VolumetricFusion(grid_size, -x_dist,x_dist, -y_dist, y_dist,-z_dist, 10.0f,
                                   trunc_val, depthIntrinsics, depthWidth, depthHeight);
        volFusion->step(depthExtrinsics, depthMat, normals, vertex_validity);
        break;
        // -------------------------------------------------------- s

//         sensor_frame += 1;
//         if (sensor_frame == 1) {
             cv::imshow("Depth Map", depth_mat);
 //            cv::imshow("Filtered Depth Map", filt_depth_mat);
//             cv::imshow("Normals Map", normalsMap_Vis);
//             waitKey(0);
//             break;
//         }

//        break;
        delete[] vertices;
        delete[] normals;
        delete volFusion;
    }

    return 0;
}