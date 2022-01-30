//
// Created by ibrahimov on 12.01.22.
//

#ifndef KINECTFUSION_POSEESTIMATION_H
#define KINECTFUSION_POSEESTIMATION_H
#endif //KINECTFUSION_POSEESTIMATION_H

#include "Eigen.h"
#include "declarations.h"
#include <iostream>

using namespace Eigen;

struct  Normal;
struct Vertex;

class PoseEstimation {
private:
//    float beta, gamma, alpha, tx, ty, tz;
//    VectorXf incrementPose = VectorXf::Zero(6,1);
//    Matrix4f transformIncrement = Matrix4f::Identity(3,4);
    const float epsDist = 0.02f;
    const float epsTheta = 0.2f;
    const int pyramid_level = 3;
    const int iters[3] = {5,5,5};

    unsigned int depthWidth, depthHeight;
    Matrix4f transform_prev;
    int* vertex_validity;
    int sensorFrame;

public:
    PoseEstimation() {}
    ~PoseEstimation() {}

    void downSampleDepthMap(float* depthMap, float* sampledDepthMap) {

        unsigned int newDepthIdx = 0;
        unsigned  int counter = 0;
        for (unsigned int i = 0; i < depthHeight; i+=2) {
            for (unsigned int j = 0; j < depthWidth; j+=2) {
                int pixel_idx = i * depthWidth + j;
                counter++;
                std::cout << "pixel idx: " << pixel_idx << std::endl;
                std::cout << "counter: " << counter << std::endl;

                //get block pixel values
                Vector4f pixel_block;
                std::vector<float> pixel_block_valid;
                float pixel_block_sum = 0.0f;
                int block_size = 0;   //non-minf vals
                if (vertex_validity[pixel_idx] == 1) {
                    pixel_block[0] = depthMap[pixel_idx];
                    pixel_block_sum += pixel_block[0];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[0]);
                } else {
                    pixel_block[0] = MINF;
                }

                if (vertex_validity[pixel_idx + 1] == 1) {
                    pixel_block[1] = depthMap[pixel_idx + 1];
                    pixel_block_sum += pixel_block[1];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[1]);
                } else {
                    pixel_block[1] = MINF;
                }

                if (vertex_validity[pixel_idx + depthWidth] == 1) {
                    pixel_block[2] = depthMap[pixel_idx + depthWidth];
                    pixel_block_sum += pixel_block[2];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[2]);
                } else {
                    pixel_block[2] = MINF;
                }

                if (vertex_validity[pixel_idx + depthWidth + 1] == 1) {
                    pixel_block[3] = depthMap[pixel_idx + depthWidth + 1];
                    pixel_block_sum += pixel_block[3];
                    block_size += 1;
                    pixel_block_valid.push_back(pixel_block[3]);
                } else {
                    pixel_block[3] = MINF;
                }

//                 calculate mean and std for non-minf vals
                if (pixel_block[0] == MINF && pixel_block[1] == MINF && pixel_block[2] == MINF && pixel_block[3] == MINF){
                    std::cout << "Skipping this..." << std::endl;
//                    sampledDepthMap[newDepthIdx] = MINF;
                    sampledDepthMap[pixel_idx] = MINF;
                    sampledDepthMap[pixel_idx + 1] = MINF;
                    sampledDepthMap[pixel_idx + depthWidth] = MINF;
                    sampledDepthMap[pixel_idx + depthWidth + 1] = MINF;
//                    newDepthIdx++;
                    continue;
                }
//
                float block_mean = (float) pixel_block_sum / block_size;
                Vector4f vec_diff;
                for (int m = 0; m < pixel_block_valid.size(); m++) {
                    vec_diff[m] = pow((pixel_block_valid[m] - block_mean),2);
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
//                sampledDepthMap[newDepthIdx] = averagedDepthVal;
//                newDepthIdx += 1;
                sampledDepthMap[pixel_idx] = averagedDepthVal;
                sampledDepthMap[pixel_idx + 1] = averagedDepthVal;
                sampledDepthMap[pixel_idx + depthWidth] = averagedDepthVal;
                sampledDepthMap[pixel_idx + depthWidth + 1] = averagedDepthVal;
            }
        }
    }

    void updateParams(unsigned int depthHeight, unsigned int depthWidth, int sensorFrame, int* vertex_validity) {
        this->depthWidth = depthWidth;
        this->depthHeight = depthHeight;
        this->vertex_validity = vertex_validity;
        this->sensorFrame = sensorFrame;
    }

    void printParams() {
        std::cout << "width: " << depthWidth << std::endl;
        std::cout << "height: " << depthHeight << std::endl;
        std::cout << "frame: " << sensorFrame << std::endl;
        for (int i = 0; i < depthHeight; i++) {
            for (int j =0; j < depthWidth; j++) {
                int curr_idx = i * depthWidth + j;
                std::cout << vertex_validity[curr_idx] << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix4f convertPoseToMatrix(Vector3f& pose) {
        Matrix4f transfer_increment;

        transfer_increment(0,0) = 1.0f;
        transfer_increment(0,1) = pose(2);
        transfer_increment(0,2) = -pose(1);
        transfer_increment(0,3) = pose(3);

        transfer_increment(1,0) = -pose(2);
        transfer_increment(1,1) = 1.0f;
        transfer_increment(1,2) = pose(0);
        transfer_increment(1,3) = pose(4);

        transfer_increment(2,0) = pose(1);
        transfer_increment(2,1) = -pose(0);
        transfer_increment(2,2) = 1.0f;
        transfer_increment(2,3) = pose(5);

        return transfer_increment;
    }

    VectorXf convertMatrixToPose(Matrix4f& transform) {
        Vector4f poseVector; // = VectorXf::Zero(6,1);
        poseVector[0] = transform(1,2);
        poseVector[1] = transform(2,0);
        poseVector[2] = transform(0,1);
        poseVector[3] = transform(0,3);
        poseVector[4] = transform(1,3);
        poseVector[5] = transform(2,3);
        return poseVector;
    }

//    Vector3f convertToGlobal(Vector4f& local_vec, Matrix4f& transform_mat) {
//       Vector3f glob_vec;
//       glob_vec = transform_mat * local_vec;
//       return glob_vec;
//    }

    Vector3f calcGlobalNormal(Vector3f& local, Matrix4f& transform) {

        Matrix3f rot_mat = transform.block(0,0,3,3);
        Vector3f glob_norm = rot_mat * local;
        return glob_norm;
    }

    Vector4f convertToLocal(Vector3f& global, Matrix4f transform) {

       MatrixXf transform_inv = transform.inverse();
       Vector3f local_temp = transform_inv * global;
       Vector4f local = Vector4f(local_temp[0], local_temp[1], local_temp[2], 1.0f);
       return local;
   }

    MatrixXf constructGmat(Vertex& vertex) {
        MatrixXf G; //= Matrix<float, 3, 6>::Zero();
        G(0,0) = 0.0f;
        G(0,1) = -vertex.pos(2);
        G(0,2) = vertex.pos(1);
        G(1,0) = vertex.pos(2);
        G(1,1) = 0.0f;
        G(1,2) = -vertex.pos(0);
        G(2,0) = -vertex.pos(1);
        G(2,1) = vertex.pos(0);
        G(2,2) = 0.0f;
        G.block(0,3, 3,3) = Matrix<float,3,3>::Identity();
        return G;
    }

    void estimatePose(Vertex* vertices, Normal* normals, Vector3f* predictedVertices, Vector3f* predictedNormals, Matrix4f& transform, Matrix3f& intrinsics) {

//        for (int level = 0; level < pyramid_level; level++) {
//            for (int iter = 0; iter < iters[level]; iter++) {
//                MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
//                MatrixXf A_transpose = Eigen::Matrix<float, 6,6>::Zero();
//                MatrixXf b = Eigen::Matrix<float, 6, 1>::Zero();
//                MatrixXf ATA = Eigen::Matrix<float, 6,6>::Zero();
//                MatrixXf ATb = Eigen::Matrix<float, 6,1>::Zero();
//
//                //loop over pixels
//                for (int i = 0; i < depthHeight; i++) {
//                    for (int j = 0; j < depthWidth; j++) {
//                        int curr_idx = i * depthWidth + j;
//                        if (vertex_validity[curr_idx] == 1) {
//                            Vertex Vg;
//                            Normal Ng;
//
//                            //calculate current normals and vertices
//                            Vg.pos = convertToGlobal(vertices[i].pos,
//                                                     transform_prev);     //current global vertices and normals.
//                            Ng.val = calcGlobalNormal(normals[i].val, transform_prev);
//
//                            //calculate prev normals and vertices
//                            Matrix4f frame_transform = transform_prev;
//                            Vector3f u_hat = intrinsics * frame_transform * vertices[i].pos;
//                            u_hat = Vector3f(u_hat[0] / u_hat[2], u_hat[1]/u_hat[2], 1.0f);
//
//                            int idx = u_hat[1] + u_hat[0] * depthWidth;
//
//                            if (predictedVertices[idx][0] == MINF) {
//                                continue;
//                            }
//                            if (predictedNormals[idx][0] == MINF) {
//                                continue;
//                            }
//
//                            //TODO check validity here
//
//                            MatrixXf G = constructGmat(Vg);
//                            A_transpose = (G.transpose() * predictedNormals[idx]);
//                            A = A_transpose.transpose();
//                            b = predsictedNormals[idx].transpose() * (predictedNormals[idx] - Vg.pos);  // TODO num of prev vertices??
//
//                            ATA += A_transpose * A;
//                            ATb += A_transpose * b;
//                        }
//                    }
//                }
//                // solve for pose vector
//                incrementPose = ATA.inverse() * ATb;
//                // convert pose vector to transform matrix
//                Matrix4f transform_increment_mat = convertPoseToMatrix(incrementPose);
//                transform = transform_increment_mat * transform_prev;
//
//                //TODO check validity here?
//
//
//            }
//        }
    }
};