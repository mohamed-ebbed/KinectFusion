//
// Created by ibrahimov on 07.02.22.
//

#ifndef KINECTFUSION2_FRAME_H
#define KINECTFUSION2_FRAME_H
#endif


#include <iostream>

#include "Eigen.h"
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include "declarations.h"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace cuda;


class Frame {

public:
    float* depthMap_L1;
    float* depthMap_L2;
    float* depthMap_L3;

    int* vertex_validity_L1 ;
    int* vertex_validity_L2 ;
    int* vertex_validity_L3 ;

    Vertex* vertices_L1;
    Vertex* vertices_L2;
    Vertex* vertices_L3;
    Normal* normals_L1 ;
    Normal* normals_L2 ;
    Normal* normals_L3 ;

    const int depthHeight_L1 = 480;
    const int depthWidth_L1  = 640;
    const int depthHeight_L2 = depthHeight_L1 / 2;
    const int depthWidth_L2  = depthWidth_L1  / 2;
    const int depthHeight_L3 = depthHeight_L2 / 2;
    const int depthWidth_L3  = depthWidth_L2  / 2;

    const unsigned int numVertices_L1 = depthHeight_L1 * depthWidth_L1;
    const unsigned int numVertices_L2 = depthHeight_L2 * depthWidth_L2;
    const unsigned int numVertices_L3 = depthHeight_L3 * depthWidth_L3;

    Frame() {
        cout << "Frame constructor called! " << endl;
        depthMap_L1 = new float[numVertices_L1];   // to be used in subsampling
        depthMap_L2 = new float[numVertices_L2];
        depthMap_L3 = new float[numVertices_L3];

        vertex_validity_L1 = new int[numVertices_L1];
        vertex_validity_L2 = new int[numVertices_L2];
        vertex_validity_L3 = new int[numVertices_L3];

        vertices_L1 = new Vertex[numVertices_L1];
        vertices_L2 = new Vertex[numVertices_L2];
        vertices_L3 = new Vertex[numVertices_L3];
        normals_L1 =  new Normal[numVertices_L1];
        normals_L2 =  new Normal[numVertices_L2];
        normals_L3 =  new Normal[numVertices_L3];
    }

    ~Frame() {
        cout << "Frame destructor called! " << endl;
        delete[] depthMap_L1;
        delete[] depthMap_L2;
        delete[] depthMap_L3;

        delete[] vertices_L1;
        delete[] vertices_L2;
        delete[] vertices_L3;

        delete[] normals_L1;
        delete[] normals_L2;
        delete[] normals_L3;

        delete[] vertex_validity_L1;
        delete[] vertex_validity_L2;
        delete[] vertex_validity_L3;
    }

    void computeVerticeLevels_cpu(Matrix3f depthIntrinsicsInv) {
        computeVertices_cpu(depthMap_L1, vertices_L1, normals_L1, depthHeight_L1, depthWidth_L1, vertex_validity_L1, depthIntrinsicsInv);
        computeVertices_cpu(depthMap_L2, vertices_L2, normals_L2, depthHeight_L2, depthWidth_L2, vertex_validity_L2, depthIntrinsicsInv);
        computeVertices_cpu(depthMap_L3, vertices_L3, normals_L3, depthHeight_L3, depthWidth_L3, vertex_validity_L3, depthIntrinsicsInv);

    }

    void computeNormalLevels_cpu(){
        computeNormals_cpu(vertices_L1, normals_L1, depthHeight_L1, depthWidth_L1);
        computeNormals_cpu(vertices_L2, normals_L2, depthHeight_L2, depthWidth_L2);
        computeNormals_cpu(vertices_L3, normals_L3, depthHeight_L3, depthWidth_L3);
    }


    void computeVertices_cpu(float* depth_mat, Vertex* vertices, Normal* normals, int depth_height, int depth_width, int* vertex_validity, Matrix3f& depthIntrinsicsInv) {
        for (unsigned int r = 0; r < depth_height; r++) {
            for (unsigned int c = 0; c < depth_width; c++) {
                unsigned int vertex_idx = r * depth_width + c;
//            float depth_pixel = depth_mat.at<float>(r, c);
                float depth_pixel = depth_mat[vertex_idx];
                normals[vertex_idx].val = Vector3f(1.0f, 1.0f, 1.0f);

                if (std::isnan(depth_pixel)) {
                    vertex_validity[vertex_idx] = 0;
                    vertices[vertex_idx].pos = Vector4f(MINF, MINF, MINF, MINF);
                } else {
                    Vector3f camera_coord = depthIntrinsicsInv * Vector3f(c, r, 1) * depth_pixel;
                    vertices[vertex_idx].pos[0] = camera_coord[0];
                    vertices[vertex_idx].pos[1] = camera_coord[1];
                    vertices[vertex_idx].pos[2] = camera_coord[2];
                    vertices[vertex_idx].pos[3] = 1.0f;
                    vertex_validity[vertex_idx] = 1;
                }
            }
        }

    }

    void computeNormals_cpu(Vertex* vertices, Normal* normals, int depth_height, int depth_width) {
        for(unsigned int v = 1; v < depth_height-1; v++) {
            for (unsigned int u = 1; u < depth_width-1 ; u++) {
                unsigned int curr_idx = u + v*depth_width;
                Vector3f v1, v2;
                Vector3f vk = Vector3f(vertices[curr_idx].pos[0], vertices[curr_idx].pos[1], vertices[curr_idx].pos[2]);
                Vector3f vk_right = Vector3f(vertices[curr_idx+1].pos[0], vertices[curr_idx+1].pos[1], vertices[curr_idx+1].pos[2]);
                Vector3f vk_up = Vector3f(vertices[curr_idx+depth_width].pos[0], vertices[curr_idx+depth_width].pos[1], vertices[curr_idx+depth_width].pos[2]);

                v1 = vk_right - vk;
                v2 = vk_up - vk;

                normals[curr_idx].val = v1.cross(v2);
                normals[curr_idx].val.normalize();
            }
        }
    }

    void subSampleDepthLevels(cv::Mat& filt_depth_mat) {
        memcpy(depthMap_L1, filt_depth_mat.data, numVertices_L1 * sizeof(float)); //copy filtered original depth map into C++ array
        subSampleDepthMap(depthMap_L1, depthMap_L2, depthHeight_L1, depthWidth_L1, vertex_validity_L1,vertex_validity_L2);
        subSampleDepthMap(depthMap_L2, depthMap_L3, depthHeight_L2, depthWidth_L2, vertex_validity_L2, vertex_validity_L3);
    }

    void subSampleDepthMap(float *depthMap, float *subSampledDepthMap, int depth_height, int depth_width, int* depth_vertex_validity, int* subDepth_vertex_validity) {

        unsigned int newDepthIdx = 0;
        for (unsigned int i = 0; i < depth_height; i += 2) {
            for (unsigned int j = 0; j < depth_width; j += 2) {
                int pixel_idx = i * depth_width + j;
                //get block pixel values
                Vector4f pixel_block;
                std::vector<float> pixel_block_valid;
                float pixel_block_sum = 0.0f;
                int block_size = 0;   //non-minf vals

                if (depth_vertex_validity[pixel_idx] == 1) {
                    pixel_block[0] = depthMap[pixel_idx];
                    pixel_block_sum += pixel_block[0];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[0]);
                } else {
                    pixel_block[0] = MINF;
                }

                if (depth_vertex_validity[pixel_idx + 1] == 1) {
                    pixel_block[1] = depthMap[pixel_idx + 1];
                    pixel_block_sum += pixel_block[1];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[1]);
                } else {
                    pixel_block[1] = MINF;
                }

                if (depth_vertex_validity[pixel_idx + depth_width] == 1) {
                    pixel_block[2] = depthMap[pixel_idx + depth_width];
                    pixel_block_sum += pixel_block[2];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[2]);
                } else {
                    pixel_block[2] = MINF;
                }

                if (depth_vertex_validity[pixel_idx + depth_width + 1] == 1) {
                    pixel_block[3] = depthMap[pixel_idx + depth_width + 1];
                    pixel_block_sum += pixel_block[3];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[3]);
                } else {
                    pixel_block[3] = MINF;
                }

//                 calculate mean and std for non-minf vals
                if (pixel_block[0] == MINF && pixel_block[1] == MINF && pixel_block[2] == MINF &&
                    pixel_block[3] == MINF) {
//                std::cout << "Skipping this..." << std::endl;
                    subSampledDepthMap[newDepthIdx] = MINF;
                    subDepth_vertex_validity[newDepthIdx] = 0;
                    newDepthIdx++;
                    continue;
                }

                float block_mean = (float) pixel_block_sum / block_size;
                subSampledDepthMap[newDepthIdx] = block_mean;
                subDepth_vertex_validity[newDepthIdx] = 1;
                newDepthIdx += 1;
            }
        }
    }

    // sub-sampling with std dev?
    void subSampleDepthMap2(float *depthMap, float *subSampledDepthMap, int depth_height, int depth_width, int* depth_vertex_validity, int* subDepth_vertex_validity) {
        unsigned int newDepthIdx = 0;
        for (unsigned int i = 0; i < depth_height; i += 2) {
            for (unsigned int j = 0; j < depth_width; j += 2) {
                int pixel_idx = i * depth_width + j;
                //get block pixel values
                Vector4f pixel_block;
                std::vector<float> pixel_block_valid;
                float pixel_block_sum = 0.0f;
                int block_size = 0;   //non-minf vals

                if (depth_vertex_validity[pixel_idx] == 1) {
                    pixel_block[0] = depthMap[pixel_idx];
                    pixel_block_sum += pixel_block[0];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[0]);
                } else {
                    pixel_block[0] = MINF;
                }

                if (depth_vertex_validity[pixel_idx + 1] == 1) {
                    pixel_block[1] = depthMap[pixel_idx + 1];
                    pixel_block_sum += pixel_block[1];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[1]);
                } else {
                    pixel_block[1] = MINF;
                }

                if (depth_vertex_validity[pixel_idx + depth_width] == 1) {
                    pixel_block[2] = depthMap[pixel_idx + depth_width];
                    pixel_block_sum += pixel_block[2];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[2]);
                } else {
                    pixel_block[2] = MINF;
                }

                if (depth_vertex_validity[pixel_idx + depth_width + 1] == 1) {
                    pixel_block[3] = depthMap[pixel_idx + depth_width + 1];
                    pixel_block_sum += pixel_block[3];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[3]);
                } else {
                    pixel_block[3] = MINF;
                }

//                 calculate mean and std for non-minf vals
                if (pixel_block[0] == MINF && pixel_block[1] == MINF && pixel_block[2] == MINF &&
                    pixel_block[3] == MINF) {
                    std::cout << "Skipping this..." << std::endl;
                    subSampledDepthMap[newDepthIdx] = MINF;
                    subDepth_vertex_validity[newDepthIdx] = 0;
                    newDepthIdx++;
                    continue;
                }
                float block_mean = (float) pixel_block_sum / block_size;
                Vector4f vec_diff;
                for (int m = 0; m < pixel_block_valid.size(); m++) {
                    vec_diff[m] = pow((pixel_block_valid[m] - block_mean), 2);
                }
                float block_std_dev_lim = 3.0f * sqrt(vec_diff.sum() / (pixel_block_valid.size() - 1));

                float block_sum = 0.0f;
                int block_idx = 0;
                for (int m = 0; m < pixel_block_valid.size(); m++) {
                    if (pixel_block_valid[m] <= block_std_dev_lim) {
                        block_sum += pixel_block_valid[m];
                        block_idx++;
                    }
                }
                float averagedDepthVal = (float) block_sum / block_idx;     // finally calculate average of valid vals
                subSampledDepthMap[newDepthIdx] = averagedDepthVal;
                subDepth_vertex_validity[newDepthIdx] = 1;
                newDepthIdx += 1;
            }
        }
    }


    void plotDepthAndNormals() {

        cv::Mat cv_depthMap_L1 = cv::Mat(static_cast<int>(depthHeight_L1), static_cast<int>(depthWidth_L1), CV_32F,depthMap_L1);
        cv::Mat cv_depthMap_L2 = cv::Mat(static_cast<int>(depthHeight_L2), static_cast<int>(depthWidth_L2), CV_32F, depthMap_L2);
        cv::Mat cv_depthMap_L3 = cv::Mat(static_cast<int>(depthHeight_L3), static_cast<int>(depthWidth_L3), CV_32F,depthMap_L3);

        cv::Mat cv_norm_L1 = cv::Mat(static_cast<int>(depthHeight_L1), static_cast<int>(depthWidth_L1), CV_32FC3,normals_L1);
        cv::Mat cv_norm_L2 = cv::Mat(static_cast<int>(depthHeight_L2), static_cast<int>(depthWidth_L2), CV_32FC3,normals_L2);
        cv::Mat cv_norm_L3 = cv::Mat(static_cast<int>(depthHeight_L3), static_cast<int>(depthWidth_L3), CV_32FC3,normals_L3);

        cv::imshow("Depth Map L1", cv_depthMap_L1);
        cv::imshow("Depth Map L2 ", cv_depthMap_L2);
        cv::imshow("Depth Map L3 ", cv_depthMap_L3);

        cv::imshow("Normal Map L1", cv_norm_L1);
        cv::imshow("Normal Map L2 ", cv_norm_L2);
        cv::imshow("Normal Map L3 ", cv_norm_L3);

        waitKey(10);
    }

    void applyBilateral(cv::Mat& depthMap, cv::Mat& filt_depthMap){
        //appy bilateralFilter to raw depth
        GpuMat filt_depth_mat_g;
        GpuMat depth_map_g;

        depth_map_g.upload(depthMap);
        cv::cuda::bilateralFilter(depth_map_g, filt_depth_mat_g, 9, 9, 0,
                                  BORDER_DEFAULT); //max d val of 5 recommended for real-time applications (9 for offline)
        filt_depth_mat_g.download(filt_depthMap);
    }


    Vertex* getVertices(int level){
        if (level == 1)
            return vertices_L1;
        if (level == 2)
            return vertices_L2;
        if (level == 3)
            return vertices_L3;
    }

    Normal* getNormals(int level){
        if (level== 1)
            return normals_L1;
        if (level == 2)
            return normals_L2;
        if (level == 3)
            return normals_L3;
    }

    int* getVertexValidity(int level){
        if (level== 1)
            return vertex_validity_L1;
        if (level == 2)
            return vertex_validity_L2;
        if (level == 3)
            return vertex_validity_L3;
    }

    unsigned int getNumVertices(int level){
        if (level== 1)
            return numVertices_L1;
        if (level == 2)
            return numVertices_L2;
        if (level == 3)
            return numVertices_L3;
    }

    int getDepthHeight(int level){
        if (level == 1)
            return depthHeight_L1;
        if (level == 2)
            return depthHeight_L2;
        if (level == 3)
            return depthHeight_L3;

    }

    int getDepthWidth(int level){
        if (level == 1)
            return depthWidth_L1;
        if (level == 2)
            return depthWidth_L2;
        if (level == 3)
            return depthWidth_L3;
    }
};