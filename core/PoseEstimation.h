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
    const Matrix<float, 1, 4> aux_vec = {0.0f, 0.0f, 0.0f, 1.0f};
    Matrix<float, 6, 1> poseVector;
    Matrix<float, 3, 4> transformIncrement;
    Matrix<float, 4, 4> transform_prev;
    Matrix<float, 4, 4> frame_transform;
    Matrix<float, 4, 4> transform;

    const float epsDist = 0.02f;
    const float epsTheta = 0.2f;
    const int pyramid_levels = 3;
    const int iters[3] = {10, 5, 4};

    unsigned int depthWidth, depthHeight;
    int *vertex_validity;
    int sensorFrame;
    Matrix3f intrinsics;  //camera intrinsics matrix

public:
    PoseEstimation() {}

    ~PoseEstimation() { std::cout << "PoseEstimation destructor called! " << std::endl; }

    void updateParams(unsigned int depthHeight, unsigned int depthWidth, int sensorFrame, int *vertex_validity,
                      Matrix3f &intrinsics) {
        this->depthWidth = depthWidth;
        this->depthHeight = depthHeight;
        this->vertex_validity = vertex_validity;
        this->sensorFrame = sensorFrame;
        this->intrinsics = intrinsics;
        if (sensorFrame == 0) {
            transform_prev = Matrix<float, 4, 4>::Identity();
            frame_transform = Matrix<float, 4, 4>::Identity();    // not used?
            transformIncrement = Matrix<float, 3, 4>::Identity();
            transformIncrement(2, 3) = 1.0f;
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

    Matrix<float, 3, 4> convertPoseToMatrix(Matrix<float, 6, 1>& pose) {
        Matrix<float, 3, 4> transfer_increment;

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


    Matrix<float,6,1> convertMatrixToPose(Matrix4f &transform) {
        Matrix<float,6,1> poseVector; // = VectorXf::Zero(6,1);
        poseVector[0] = transform(1, 2);
        poseVector[1] = transform(2, 0);
        poseVector[2] = transform(0, 1);
        poseVector[3] = transform(0, 3);
        poseVector[4] = transform(1, 3);
        poseVector[5] = transform(2, 3);
        return poseVector;
    }

    Vector3f convertToGlobal(Vector4f& local_vec, Matrix<float, 4, 4>& transform_mat) {
        Vector4f glob_vec_temp = transform_mat * local_vec;
        Vector3f glob_vec = Vector3f(glob_vec_temp[0], glob_vec_temp[1], glob_vec_temp[2]);
        return glob_vec;
    }

    Vector3f calcGlobalNormal(Vector3f& local, Matrix<float, 4, 4>& transform) {
        Matrix<float, 3, 3> rot_mat = transform.block(0, 0, 3, 3);
        Vector3f glob_norm = rot_mat * local;
        return glob_norm;
    }

    Vector4f convertToLocal(Vector3f& global, Matrix4f transform) {
        MatrixXf transform_inv = transform.inverse();
        Vector3f local_temp = transform_inv * global;
        Vector4f local = Vector4f(local_temp[0], local_temp[1], local_temp[2], 1.0f);
        return local;
    }

    Matrix4f getTransform() {
        return transform;
    }

    Matrix<float, 3, 6> constructGmat(Vector3f vec) {
        Matrix<float, 3, 6> G;
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

    Matrix<float, 4, 4> estimatePose2(Vertex *vertices, Normal *normals, Vector3f *predictedVertices, Vector3f *predictedNormals) {

        cout << "inside estimatePose" << endl;

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
                    u_hat = Vector3f(u_hat[0] / u_hat[2], u_hat[1] / u_hat[2], 1.0f);
                    unsigned int u_idx = u_hat[0] + u_hat[1] * depthWidth;
//                    cout << "u_hat: "<< "[ " << u_hat[0] << " , " <<  u_hat[1] << " ]" << endl;

                    if (predictedVertices[u_idx][0] == MINF) { continue; }
                    if (predictedNormals[u_idx][0] == MINF) { continue; }
                    if (isnan(u_idx) || u_idx < 0 || u_idx > depthWidth || u_idx > depthHeight) { continue; }


                    Matrix<float, 3, 6> G = constructGmat(point3d.pos);
                    A_transpose = (G.transpose() * predictedNormals[u_idx]);
//                    cout << "Predicted_normals: " << endl << predictedNormals[u_idx] << endl;
                    A = A_transpose.transpose();
                    b = predictedNormals[u_idx].transpose() *
                        (predictedVertices[u_idx] - point3d.pos);  // TODO num of prev vertices??

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
        Matrix<float, 3, 4> new_transform = transformIncrement * transform_prev;
        transform.block(0, 0, 3, 4) = new_transform;
        transform.block(3, 0, 1, 4) = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);

        return transform;
    }


    Matrix4f estimatePose(Vertex *vertices, Normal *normals, Vector3f *predictedVertices, Vector3f *predictedNormals,
                          int *vertex_validity, Matrix4f &previousPose, Matrix3f &intrinsics, float depthWidth,
                          float depthHeight) {

        MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
        MatrixXf A_transpose = Eigen::Matrix<float, 6, 6>::Zero();
        MatrixXf b = Eigen::Matrix<float, 1, 1>::Zero();
        MatrixXf ATA = Eigen::Matrix<float, 6, 6>::Zero();
        MatrixXf ATb = Eigen::Matrix<float, 6, 1>::Zero();
        MatrixXf previousPoseInv = previousPose.inverse();
        MatrixXf pose;

        MatrixXf curr_transform = previousPose;

        for (int iter = 0; iter < 1; iter++) {

            //loop over pixels
            for (int i = 0; i < depthHeight; i++) {
                for (int j = 0; j < depthWidth; j++) {
                    int curr_idx = i * depthWidth + j;
                    if (vertex_validity[curr_idx] == 1) {
                        Vertex Vg;
                        Normal Ng;
                        //calculate current normals and vertices
                        Vg.pos = curr_transform * vertices[curr_idx].pos;
                        Ng.val = curr_transform.block(0, 0, 3, 3) * normals[curr_idx].val;

                        //calculate prev normals and vertices
                        Matrix4f frame_transform = previousPoseInv * curr_transform;
                        Vector4f v_c = frame_transform * Vector4f(vertices[curr_idx].pos[0], vertices[curr_idx].pos[1],
                                                                  vertices[curr_idx].pos[2], 1);
                        Vector3f v_c3d = Vector3f(v_c[0], v_c[1], v_c[2]);

                        Vector3f u_hat = intrinsics * v_c3d;
                        u_hat = Vector3f(u_hat[0] / u_hat[2], u_hat[1] / u_hat[2], 1.0f);
                        int idx = u_hat[0] + u_hat[1] * depthWidth;

                        if (idx <= 0 || idx >= depthWidth * depthHeight)
                            continue;


                        if (predictedVertices[idx][0] == MINF) {
                            continue;
                        }

                        Matrix3f currRot = curr_transform.block(0, 0, 3, 3);

                        float distance = (curr_transform * vertices[curr_idx].pos - predictedVertices[idx]).norm();
                        float normalSim = ((currRot * normals[curr_idx].val).dot(predictedNormals[idx]));

                        if (isnan(normalSim) || isnan(distance))
                            continue;

                        if (distance >= 0.005 || normalSim < 0.9) {
                            continue;
                        }


                        MatrixXf G = Eigen::Matrix<float, 6, 6>::Zero();
                        G(0, 0) = 0.0f;
                        G(0, 1) = -Vg.pos(2);
                        G(0, 2) = Vg.pos(1);
                        G(1, 0) = Vg.pos(2);
                        G(1, 1) = 0.0f;
                        G(1, 2) = -Vg.pos(0);
                        G(2, 0) = -Vg.pos(1);
                        G(2, 1) = Vg.pos(0);
                        G(2, 2) = 0.0f;
                        G.block(0, 3, 3, 3) = Matrix<float, 3, 3>::Identity();

                        A_transpose = (G.transpose() * predictedNormals[idx]);

                        A = A_transpose.transpose();
                        b = predictedNormals[idx].transpose() *
                            (predictedNormals[idx] - Vector3f(Vg.pos[0], Vg.pos[1], Vg.pos[2]));

                        ATA += A_transpose * A;
                        ATb += A_transpose * b;
                    }
                }
            }
            // solve for pose vector
            pose = ATA.inverse() * ATb;
            // convert pose vector to transform matrix
            Matrix4f transfer_increment = Matrix4f::Identity();

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

//        cout << "calculated pose" << endl;
//        cout << curr_transform << endl;
            curr_transform = transfer_increment * curr_transform;

            cout << curr_transform << endl;

        }
        return curr_transform;
    }



    Matrix<float,4,4> estimatePose_levels(Frame& frame, Vertex *vertices, Normal *normals, Vector3f *predictedVertices, Vector3f *predictedNormals,
                                       int *vertex_validity, Matrix4f &previousPose, Matrix3f &intrinsics, float depthWidth,
                                       float depthHeight) {
        MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
        MatrixXf A_transpose = Eigen::Matrix<float, 6, 6>::Zero();
        MatrixXf b = Eigen::Matrix<float, 1, 1>::Zero();
        MatrixXf ATA = Eigen::Matrix<float, 6, 6>::Zero();
        MatrixXf ATb = Eigen::Matrix<float, 6, 1>::Zero();
        MatrixXf previousPoseInv = previousPose.inverse();
        MatrixXf pose;

        MatrixXf curr_transform = previousPose;



        for (int level = 0; level < pyramid_levels; level++) {

            //fetch corresponding vertices/normals from frame object
            Vertex* vertices = new Vertex[frame.getNumVertices(level)];
            Normal* normals = new Normal[frame.getNumVertices(level)];
            int* vertex_validity = new int[frame.getNumVertices(level)];

            vertices = frame.getVertices(level);
            normals = frame.getNormals(level);
            vertex_validity = frame.getVertexValidity(level);

            const int depth_height = frame.getDepthHeight(level);
            const int depth_width  = frame.getDepthWidth(level);


            for (int iter = 0; iter < iters[level]; iter++) {

                for (int i = 0; i < depth_height; i++) {
                    for (int j = 0; j < depth_width; j++) {
                        int curr_idx = i * depth_width + j;
                        if (vertex_validity[curr_idx] == 1) {
                            Vertex Vg;
                            Normal Ng;
                            //calculate current normals and vertices
                            Vg.pos = curr_transform * vertices[curr_idx].pos;
                            Ng.val = curr_transform.block(0, 0, 3, 3) * normals[curr_idx].val;

                            //calculate prev normals and vertices
                            Matrix4f frame_transform = previousPoseInv * curr_transform;
                            Vector4f v_c =
                                    frame_transform * Vector4f(vertices[curr_idx].pos[0], vertices[curr_idx].pos[1],
                                                               vertices[curr_idx].pos[2], 1);
                            Vector3f v_c3d = Vector3f(v_c[0], v_c[1], v_c[2]);

                            Vector3f u_hat = intrinsics * v_c3d;
                            u_hat = Vector3f(u_hat[0] / u_hat[2], u_hat[1] / u_hat[2], 1.0f);
                            int idx = u_hat[0] + u_hat[1] * depthWidth;

                            if (idx <= 0 || idx >= depthWidth * depthHeight)
                                continue;


                            if (predictedVertices[idx][0] == MINF) {
                                continue;
                            }

                            Matrix3f currRot = curr_transform.block(0, 0, 3, 3);

                            float distance = (curr_transform * vertices[curr_idx].pos - predictedVertices[idx]).norm();
                            float normalSim = ((currRot * normals[curr_idx].val).dot(predictedNormals[idx]));

                            if (isnan(normalSim) || isnan(distance))
                                continue;

                            if (distance >= 0.005 || normalSim < 0.9) {
                                continue;
                            }

                            MatrixXf G = Eigen::Matrix<float, 6, 6>::Zero();
                            G(0, 0) = 0.0f;
                            G(0, 1) = -Vg.pos(2);
                            G(0, 2) = Vg.pos(1);
                            G(1, 0) = Vg.pos(2);
                            G(1, 1) = 0.0f;
                            G(1, 2) = -Vg.pos(0);
                            G(2, 0) = -Vg.pos(1);
                            G(2, 1) = Vg.pos(0);
                            G(2, 2) = 0.0f;
                            G.block(0, 3, 3, 3) = Matrix<float, 3, 3>::Identity();

                            A_transpose = (G.transpose() * predictedNormals[idx]);

                            A = A_transpose.transpose();
                            b = predictedNormals[idx].transpose() *
                                (predictedNormals[idx] - Vector3f(Vg.pos[0], Vg.pos[1], Vg.pos[2]));

                            ATA += A_transpose * A;
                            ATb += A_transpose * b;
                        }
                    }
                }
                // solve for pose vector
                pose = ATA.inverse() * ATb;
                // convert pose vector to transform matrix
                Matrix4f transfer_increment = Matrix4f::Identity();

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

                curr_transform = transfer_increment * curr_transform;
                cout << curr_transform << endl;
            }

            //manual garbage collection..
            delete[] vertex_validity;
            delete[] normals;
            delete[] vertices;
        }
        return curr_transform;
    }
};