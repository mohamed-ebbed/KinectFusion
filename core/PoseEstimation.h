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
using namespace std;

//struct  Normal;
//struct Vertex;

class PoseEstimation {
private:
    const Matrix<float, 1, 4> aux_vec = {0.0f, 0.0f,0.0f, 1.0f};
    Matrix<float, 6, 1> poseVector;
    Matrix<float, 3,4> transformIncrement;
    Matrix<float,4,4> transform_prev;
    Matrix<float, 4,4> frame_transform;
    Matrix<float,4,4> transform;

    const float epsDist = 0.02f;
    const float epsTheta = 0.2f;
    const int pyramid_level = 3;
    const int iters[3] = {5, 5, 5};

    unsigned int depthWidth, depthHeight;
    int *vertex_validity;
    int sensorFrame;
    Matrix3f intrinsics;  //camera intrinsics matrix

public:
    PoseEstimation() {}
    ~PoseEstimation() { std::cout << "PoseEstimation destructor called! " << std::endl; }

    void downSampleDepthMap(float *depthMap, float *sampledDepthMap) {
        unsigned int newDepthIdx = 0;
        unsigned int counter = 0;
        for (unsigned int i = 0; i < depthHeight; i += 2) {
            for (unsigned int j = 0; j < depthWidth; j += 2) {
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
                if (pixel_block[0] == MINF && pixel_block[1] == MINF && pixel_block[2] == MINF &&
                    pixel_block[3] == MINF) {
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
//                sampledDepthMap[newDepthIdx] = averagedDepthVal;
//                newDepthIdx += 1;
                sampledDepthMap[pixel_idx] = averagedDepthVal;
                sampledDepthMap[pixel_idx + 1] = averagedDepthVal;
                sampledDepthMap[pixel_idx + depthWidth] = averagedDepthVal;
                sampledDepthMap[pixel_idx + depthWidth + 1] = averagedDepthVal;
            }
        }
    }

    void updateParams( unsigned int depthHeight, unsigned int depthWidth, int sensorFrame, int *vertex_validity, Matrix3f& intrinsics) {
        this->depthWidth = depthWidth;
        this->depthHeight = depthHeight;
        this->vertex_validity = vertex_validity;
        this->sensorFrame = sensorFrame;
        this->intrinsics = intrinsics;
        if (sensorFrame == 0) {
            transform_prev = Matrix<float,4,4>::Identity();
            frame_transform = Matrix<float,4,4>::Identity();    // not used?
            transformIncrement = Matrix<float,3,4>::Identity();
            transformIncrement(2,3) = 1.0f;
        } else {
            cout << "No need to initialize in this frame" << endl;
        }
    }

    void printParams() {
        std::cout << "width: " << depthWidth << std::endl;
        std::cout << "height: " << depthHeight << std::endl;
        std::cout << "frame: " << sensorFrame << std::endl;
        for (int i = 0; i < depthHeight; i++) {
            for (int j = 0; j < depthWidth; j++) {
                int curr_idx = i * depthWidth + j;
                std::cout << vertex_validity[curr_idx] << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix<float,3,4> convertPoseToMatrix(Matrix<float, 6,1>& pose) {
        Matrix<float,3,4> transfer_increment;

        transfer_increment(0, 0) = 1.0f;
        transfer_increment(0, 1) = pose(2);
        transfer_increment(0, 2) = -pose(1);
        transfer_increment(0, 3) = pose(3);

        transfer_increment(1, 0) = -pose(2);
        transfer_increment(1, 1) = 1.0f;
        transfer_increment(1, 2) = pose(0);
        transfer_increment(1, 3) = pose(4);

        transfer_increment(2, 0) = pose(1);
        transfer_increment(2, 1) = -pose(0);
        transfer_increment(2, 2) = 1.0f;
        transfer_increment(2, 3) = pose(5);

        return transfer_increment;
    }

    VectorXf convertMatrixToPose(Matrix4f &transform) {
        Vector4f poseVector; // = VectorXf::Zero(6,1);
        poseVector[0] = transform(1, 2);
        poseVector[1] = transform(2, 0);
        poseVector[2] = transform(0, 1);
        poseVector[3] = transform(0, 3);
        poseVector[4] = transform(1, 3);
        poseVector[5] = transform(2, 3);
        return poseVector;
    }

    Vector3f convertToGlobal(Vector4f& local_vec, Matrix<float,4,4>& transform_mat) {
        Vector4f glob_vec_temp = transform_mat * local_vec;
        Vector3f glob_vec = Vector3f(glob_vec_temp[0], glob_vec_temp[1], glob_vec_temp[2]);
        return glob_vec;
    }

    Vector3f calcGlobalNormal(Vector3f& local, Matrix<float,4,4>& transform) {
        Matrix<float,3,3> rot_mat = transform.block(0, 0, 3, 3);
        Vector3f glob_norm = rot_mat * local;
        return glob_norm;
    }

    Vector4f convertToLocal(Vector3f &global, Matrix4f transform) {
        MatrixXf transform_inv = transform.inverse();
        Vector3f local_temp = transform_inv * global;
        Vector4f local = Vector4f(local_temp[0], local_temp[1], local_temp[2], 1.0f);
        return local;
    }

    Matrix<float,4,4> getTransform() {
        return transform;
    }

    Matrix<float, 3,6> constructGmat(Vector3f vec) {
        Matrix<float, 3,6> G;
        G(0, 0) = 0.0f;
        G(0, 1) = -vec(2);
        G(0, 2) = vec(1);
        G(1, 0) = vec(2);
        G(1, 1) = 0.0f;
        G(1, 2) = -vec(0);
        G(2, 0) = -vec(1);
        G(2, 1) = vec(0);
        G(2, 2) = 0.0f;
        G.block(0, 3, 3, 3) = Matrix<float, 3, 3>::Identity();
        return G;
    }

    Matrix<float,4,4> estimatePose(Vertex* vertices, Normal* normals, Vector3f* predictedVertices, Vector3f* predictedNormals) {

        cout << "inside estimatePose" << endl;
//        for (int level = 0; level < pyramid_level; level++) {
//            for (int iter = 0; iter < iters[level]; iter++) {

        MatrixXf A = Matrix<float, 6, 6>::Zero();
        MatrixXf A_transpose = Matrix<float, 6, 6>::Zero();
        MatrixXf b = Matrix<float, 1, 1>::Zero();
        MatrixXf ATA = Matrix<float, 6, 6>::Zero();
        MatrixXf ATb = Matrix<float, 6, 1>::Zero();

        //loop over pixels
        for (int i = 0; i < depthHeight; i++) {
            for (int j = 0; j < depthWidth; j++) {
                unsigned int curr_idx = i * depthWidth + j;
                if (vertex_validity[curr_idx] == 1) {
                    Point3D point3d;
                    point3d.pos = convertToGlobal(vertices[curr_idx].pos, transform_prev);     //current global vertices
                    point3d.normal = calcGlobalNormal(normals[curr_idx].val, transform_prev);

                    Vector3f u_hat = intrinsics * transformIncrement * vertices[curr_idx].pos;
                    u_hat = Vector3f(u_hat[0] / u_hat[2], u_hat[1]/u_hat[2], 1.0f);
                    unsigned int u_idx = u_hat[0] + u_hat[1] * depthWidth;
//                    cout << "u_hat: "<< "[ " << u_hat[0] << " , " <<  u_hat[1] << " ]" << endl;

                    if (predictedVertices[u_idx][0] == MINF) { continue; }
                    if (predictedNormals[u_idx][0] == MINF) { continue; }
                    if (isnan(u_idx) || u_idx < 0 || u_idx > depthWidth || u_idx > depthHeight) { continue; }


                    Matrix<float,3,6> G = constructGmat(point3d.pos);
                    A_transpose = (G.transpose() * predictedNormals[u_idx]);
//                    cout << "Predicted_normals: " << endl << predictedNormals[u_idx] << endl;
                    A = A_transpose.transpose();
                    b = predictedNormals[u_idx].transpose() * (predictedVertices[u_idx] - point3d.pos);  // TODO num of prev vertices??

                    ATA += A_transpose * A;
                    ATb += A_transpose * b;

                }
            }
        }

        cout << "ATA: " << endl << ATA << endl;
        cout << "ATb: " << endl << ATb << endl;

//        poseVector = ATA.inverse() * ATb;
        poseVector = ATA.fullPivLu().solve(ATb);
//        cout << "poseVector: " << endl << poseVector << endl;

        transformIncrement = convertPoseToMatrix(poseVector);
        Matrix<float, 3,4> new_transform = transformIncrement * transform_prev;
        transform.block(0,0, 3,4) = new_transform;
        transform.block(3,0,1,4) = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);

//        cout << "finished estimatePose" << endl;
        return transform;
    }



    Matrix4f estimatePose_new(Vertex* vertices, Normal* normals, Vector3f* predictedVertices, Vector3f* predictedNormals, int* vertex_validity, Matrix4f& previousPose,
                              Matrix3f& intrinsics, float depthWidth, float depthHeight) {

        Matrix4f curr_transform = previousPose;
//        cout << "Curr_transform: " << previousPose << endl;

        for (int iter = 0; iter < 1; iter++) {

            MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
            MatrixXf A_transpose = Eigen::Matrix<float, 6,6>::Zero();
            MatrixXf b = Eigen::Matrix<float, 1, 1>::Zero();
            MatrixXf ATA = Eigen::Matrix<float, 6,6>::Zero();
            MatrixXf ATb = Eigen::Matrix<float, 6,1>::Zero();
            MatrixXf previousPoseInv = previousPose.inverse();
            Matrix<float, 6,1> pose;

            cout << "before for loop" << endl;
            //loop over pixels
            for (int i = 0; i < depthHeight; i++) {
                for (int j = 0; j < depthWidth; j++) {
                    int curr_idx = i * depthWidth + j;
                    if (vertex_validity[curr_idx] == 1) {
                        Point3D point3D;
                        Vector4f temp_pos = previousPose * vertices[curr_idx].pos;
                        point3D.pos = Vector3f(temp_pos[0], temp_pos[1], temp_pos[2]);
                        point3D.normal = previousPose.block(0, 0, 3, 3) * normals[curr_idx].val;
//                        cout << "point3d calculated" << endl;

                        //calculate prev normals and vertices?
                        Matrix4f frame_transform = previousPoseInv * curr_transform;
//                        cout << "frame_transform: " << frame_transform << endl;

                        Vector4f v_c = frame_transform * Vector4f(vertices[curr_idx].pos[0], vertices[curr_idx].pos[1],
                                                                  vertices[curr_idx].pos[2], 1.0f);
                        Vector3f v_c3d = Vector3f(v_c[0], v_c[1], v_c[2]);

//                        cout << "v_c3d: " << v_c3d << endl;
                        Vector3f u_hat_temp = intrinsics * v_c3d;
                        Vector3i u_hat = Vector3f(round(u_hat_temp[0] / u_hat_temp[2]), round(u_hat_temp[1] / u_hat_temp[2]), 1);
//                        cout << "u_hat calculated" << endl;

                        unsigned int idx = u_hat[0] + u_hat[1] * depthWidth;  //TODO which one here?
//                        int idx = u_hat[1] + u_hat[0] * depthWidth;

//                        cout << "u_hat: " << u_hat << endl;
//                        cout << "u_hat idx: " << idx << endl;
//                        cout << predictedVertices[idx] << endl;

                        if (predictedVertices[idx][0] == MINF) {
                            continue;
                        }
                        if (predictedNormals[idx][0] == MINF) {
                            continue;
                        }

//                        cout << "before G mat" << endl;
                        Matrix<float, 3,6> G = Matrix<float, 3,6>::Zero(); // init G mat to zeros
                        G(0, 0) = 0.0f;
                        G(0, 1) = -point3D.pos(2);
                        G(0, 2) = point3D.pos(1);
                        G(1, 0) = point3D.pos(2);
                        G(1, 1) = 0.0f;
                        G(1, 2) = -point3D.pos(0);
                        G(2, 0) = -point3D.pos(1);
                        G(2, 1) = point3D.pos(0);
                        G(2, 2) = 0.0f;
//                        cout << "inside G mat" << endl;
                        G.block(0, 3, 3, 3) = Matrix<float, 3, 3>::Identity();
//                        cout << "G = " << G << endl;
//                        cout << "G calculated" << endl;

                        A_transpose = (G.transpose() * predictedNormals[idx]);
                        A = A_transpose.transpose();
                        b = predictedNormals[idx].transpose() * (predictedVertices[idx] - point3D.pos);
//                        cout << "A and b calculated"<< endl;
                        ATA += A_transpose * A;
                        ATb += A_transpose * b;

                    }
                }
            }

            // solve for pose
            pose = ATA.inverse() * ATb;
            // convert pose to transform_increment
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

            curr_transform = transfer_increment * curr_transform;
            cout << "transform size: " << curr_transform.size() << endl;
            cout << "End of loop" << endl;
        }
        cout << "End of estimatePose_new" << endl;
        return curr_transform;
    }

    Matrix4f estimatePose2(Vertex* vertices, Normal* normals, Vector3f* predictedVertices, Vector3f* predictedNormals, int* vertex_validity, Matrix4f& previousPose,
                           Matrix3f& intrinsics, float depthWidth, float depthHeight) {

        MatrixXf curr_transform = previousPose;
        for (int iter = 0; iter < 1; iter++) {
            MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
            MatrixXf A_transpose = Eigen::Matrix<float, 6,6>::Zero();
            MatrixXf b = Eigen::Matrix<float, 1, 1>::Zero();
            MatrixXf ATA = Eigen::Matrix<float, 6,6>::Zero();
            MatrixXf ATb = Eigen::Matrix<float, 6,1>::Zero();
            MatrixXf previousPoseInv = previousPose.inverse();
            MatrixXf pose;
//            cout << "before inner for loop" << endl;

            //loop over pixels
            for (int i = 0; i < depthHeight; i++) {
                for (int j = 0; j < depthWidth; j++) {
                    int curr_idx = i * depthWidth + j;
                    if (vertex_validity[curr_idx] == 1) {
                        Vertex Vg;
                        Normal Ng;

                        //calculate current normals and vertices
                        Vg.pos = previousPose * vertices[i].pos;
                        Ng.val = previousPose.block(0,0,3,3) * normals[i].val;

                        //calculate prev normals and vertices
                        Matrix4f frame_transform = previousPoseInv * curr_transform;
                        Vector4f v_c = frame_transform * Vector4f(vertices[i].pos[0], vertices[i].pos[1], vertices[i].pos[2], 1);
                        Vector3f v_c3d = Vector3f(v_c[0], v_c[1], v_c[2]);
                        Vector3f u_hat_temp = intrinsics * v_c3d;
                        Vector3i u_hat = Vector3i(round(u_hat_temp[0] / u_hat_temp[2]), round(u_hat_temp[1]/u_hat_temp[2]), 1);

                        unsigned int idx = u_hat[0] + u_hat[1] * depthWidth;

                        if (predictedVertices[idx][0] == MINF) {
                            continue;
                        }
                        if (predictedNormals[idx][0] == MINF) {
                            continue;
                        }

                        MatrixXf G;
                        G(0,0) = 0.0f;
                        G(0,1) = -Vg.pos(2);
                        G(0,2) = Vg.pos(1);
                        G(1,0) = Vg.pos(2);
                        G(1,1) = 0.0f;
                        G(1,2) = -Vg.pos(0);
                        G(2,0) = -Vg.pos(1);
                        G(2,1) = Vg.pos(0);
                        G(2,2) = 0.0f;

                        G.block(0,3,3,3) = Matrix<float,3,3>::Identity();

                        A_transpose = (G.transpose() * predictedNormals[idx]);
                        A = A_transpose.transpose();
                        b = predictedNormals[idx].transpose() * (predictedVertices[idx] - Vector3f(Vg.pos[0], Vg.pos[1], Vg.pos[2]));  // TODO num of prev vertices??

                        ATA += A_transpose * A;
                        ATb += A_transpose * b;
                    }
                }
            }
            // solve for pose vector
            pose = ATA.inverse() * ATb;
            // convert pose vector to transform matrix
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

            curr_transform = transfer_increment * curr_transform;
            curr_transform.block(3,0,1,4) = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
        }

//        cout << "End of pose func" << endl;
        return curr_transform;
    }
};

