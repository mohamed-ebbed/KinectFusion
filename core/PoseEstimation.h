//
// Created by ibrahimov on 12.01.22.
//

#ifndef KINECTFUSION_POSEESTIMATION_H
#define KINECTFUSION_POSEESTIMATION_H
#endif //KINECTFUSION_POSEESTIMATION_H

#include "Eigen.h"
#include "declarations.h"

using namespace Eigen;

struct  Normal;
struct Vertex;

class PoseEstimation {
private:
    float beta, gamma, alpha, tx, ty, tz;
    VectorXf incrementPose = VectorXf::Zero(6,1);
    Matrix4f transformIncrement = Matrix4f::Identity(3,4);
    const float epsDist = 0.01f; //arbitrary?
    const float epsTheta = 0.1f;
    const int pyramid_level = 3;
    const int iters[3] = {10,5,4};

    unsigned int depthWidth, depthHeight;
    Matrix4f transform_prev = Matrix4f::Zero(3,4);

    int* vertex_validity;
    unsigned int numValidPixels;
    bool isValidCorrespondence = false;

public: 

    PoseEstimation() {

    }
    ~PoseEstimation() { std::cout << "PoseEstimation object destroyed!" << std::endl; }

    void updateParams(unsigned int depthHeight, unsigned int depthWidth, int* vertex_validity, unsigned int numValidPixels) {
        this->depthWidth = depthWidth;
        this->depthHeight = depthHeight;
        this->numValidPixels = numValidPixels;
        for (unsigned int i =0; i < depthWidth * depthHeight; i++) {
            this->vertex_validity[i] = vertex_validity[i];
        }
    }

    static Matrix4f convertPoseToMatrix(VectorXf& pose) {

        Matrix4f transfer_increment = Matrix<float, 3,4>::Zero();
        
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

    static VectorXf convertMatrixToPose(Matrix4f& transform_increment) {
        Vector4f poseVector = VectorXf::Zero(6,1);
        poseVector[0] = transform_increment[1,2];
        poseVector[1] = transform_increment[2,0];
        poseVector[2] = transform_increment[0,1];
        poseVector[3] = transform_increment[0,3];
        poseVector[4] = transform_increment[1,3];
        poseVector[5] = transform_increment[2,3];
        return poseVector;
    }

   static Vector3f convertToGlobal(Vector4f& local, Matrix4f& transform) {
       Vector3f glob = transform * local;
//       Vector3f glob = Vector3f (glob_temp[0], glob_temp[1], glob_temp[2]);
       return glob;
 }

    static Vector3f calcGlobalNormal(Vector3f& local, Matrix4f& transform) {

        Matrix3f rot = transform.block(0,0,3,3);
        Vector3f glob_norm = rot * local;
        return glob_norm;
    }
   static Vector4f convertToLocal(Vector3f& global, Matrix4f transform) {

       MatrixXf transform_inv = transform.inverse();
       Vector3f local_temp = transform_inv * global;
       Vector4f local = Vector4f(local_temp[0], local_temp[1], local_temp[2], 1.0f);
       return local;
   }

    static MatrixXf constructGmat(Vertex& vertex) {
        MatrixXf G = Matrix<float, 3, 6>::Zero();
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

    void estimatePose(Vertex* vertices, Normal* normals, Vertex& prevGlobVertices, Normal& prevGlobNormals, Matrix4f& transform) {

        Eigen::Matrix3f rotation = transform.block(0, 0, 3, 3);
        Eigen::Vector3f translation = transform.block(0, 3, 3, 1);
        Eigen::Matrix3f rotation_prev = transform_prev.block(0, 0, 3, 3);
        Eigen::Vector3f translation_prev = transform_prev.block(0, 3, 3, 1);

        for (int level = 0; level < pyramid_level; level++) {
            for (int iter = 0; iter < iters[level]; iter++) {

                for (int i = 0; i < depthHeight; i++) {
                    for (int j = 0; j < depthWidth; j++) {
                        int curr_idx = i * depthWidth + j;
                        if (vertex_validity[curr_idx] == 1) {
                            MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
                            MatrixXf b = Eigen::Matrix<float, 6, 1>::Zero();
                            MatrixXf G = Eigen::Matrix<float, 3, 6>::Zero();

                            Vertex Vg;
                            Normal Ng;
                            Vg.pos = convertToGlobal(vertices[i].pos,
                                                     transform_prev);     //current global vertices and normals.
                            Ng.val = calcGlobalNormal(normals[i].val, transform_prev);

                            // TODO calculate vertex distances and normal angles


                            if (isValidCorrespondence) {
                                G = constructGmat(Vg);
                                A = (G.transpose() * prevGlobNormals.val).transpose();
                                b = prevGlobNormals.val.transpose() * (prevGlobVertices.pos - Vg.pos);

                                // solve for pose vector
                                incrementPose = (A.transpose() * A).inverse() * A.transpose() * b;
                                // convert pose vector to transform matrix
                                Matrix4f transform_increment_mat = convertPoseToMatrix(incrementPose);
                                transform = transform_increment_mat * transform_prev.inverse(); // TODO check again

                                transform_prev = transform;  // update transform for the next iter
                            }
                            else
                            {
                                continue;
                            }

                        }
                    }
                }
            }
        }
    }
};