//
// Created by ibrahimov on 12.01.22.
//

#ifndef KINECTFUSION_POSEESTIMATION_H
#define KINECTFUSION_POSEESTIMATION_H
#endif //KINECTFUSION_POSEESTIMATION_H

#include "Eigen.h"

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

public:
    PoseEstimation() {}
    ~PoseEstimation() {}

    Matrix4f convertPoseToMatrix(VectorXf& pose) {
        Matrix4f transfer_increment = Matrix<float, 3,4>::Zero();

        transfer_increment(0,0) = 1.0f;
        transfer_increment(0,1) = pose(1);
        transfer_increment(0,2) = -pose(2);
        transfer_increment(0,3) = pose(4);
        transfer_increment(1,0) = -pose(0);
        transfer_increment(1,2) = pose(1);
        transfer_increment(1,3) = pose(5);
        transfer_increment(2,1) = pose(1);
        //TODO finish the function
    }
    void convertMatrixToPose(Matrix4f& transform, VectorXf& pose) {}

    static MatrixXf convertToGlobal(MatrixXf& local, Matrix4f& transform) {
        MatrixXf global;
//        global = transform * local;
        for (unsigned int i = 0; i < local.rows(); i++) {
            Vector4f current_local(local(i,0), local(i,1), local(i,2), 1);
            Vector3f current_global = transform * current_local;
            global.block(i,0,1,3) = current_global;
        }
        return global;  // can be normals or vertices etc.
    }

    static MatrixXf convertToLocal(MatrixXf global, Matrix4f transform) {
        MatrixXf local;
        MatrixXf transform_inv = transform.inverse();
        for (unsigned int i = 0; i < global.rows(); i++) {
            Vector3f current_global(global(i,0), global(i,1), global(i,2));
            Vector4f current_local = transform_inv * current_global;
            local.block(i,0,1,3) = current_local;
            local(i, 3) = 1.0f;
        }
        return local;
    }

    static MatrixXf constructGmat(Vector3f& vertex) {
        MatrixXf G = Matrix<float, 3, 6>::Zero();
        G(0,0) = 0.0f;
        G(0,1) = -vertex(2);
        G(0,2) = vertex(1);
        G(1,0) = vertex(2);
        G(1,1) = 0.0f;
        G(1,2) = -vertex(0);
        G(2,0) = -vertex(3);
        G(2,1) = vertex(0);
        G(2,2) = 0.0f;
        G.block(0,3, 3,3) = Matrix<float,3,3>::Identity();
        return G;
    }

    void estimatePose(MatrixXf& vertices, MatrixXf& prevVertices, MatrixXf& normals, MatrixXf& prevNormals, Matrix4f& transform, Matrix4f& transform_prev) {

        Eigen::Matrix3f rotation = transform.block(0,0,3,3);
        Eigen::Vector3f translation = transform.block(0,3,3,1);
        Eigen::Matrix3f rotation_prev = transform_prev.block(0,0,3,3);
        Eigen::Vector3f translation_prev = transform_prev.block(0,3,3,1);

        // do estimation from coarse to fine vertex sampling
        for (int level = 0; level < pyramid_level; level++) {
            // different iterations for each levels
            for(int iter= 0; iter < iters[level]; iter++) {
                Eigen::MatrixXf A = Eigen::Matrix<float,6,6>::Zero();
                Eigen::MatrixXf b = Eigen::Matrix<float,6,1>::Zero();
                Eigen::MatrixXf G = Eigen::Matrix<float,3,6>::Zero();

                bool isValidEstimation = false;
                assert(normals.rows() ==  vertices.rows());  //check normal-vertex correspondence?
                assert(prevNormals.rows() == prevVertices.rows()) ;

                Vector3f global_vertices = convertToGlobal(vertices, transform_prev);
                Vector3f global_normals = convertToGlobal(normals, transform_prev);
                Vector3f global_prevVertices = global_vertices;
                Vector3f global_prevNormals = global_normals;

                for (unsigned int vertex = 0; vertex < vertices.rows(); vertex++) {
                    //convert normals and vertices to global
                    Vector3f glob_vertex = global_vertices.block(vertex,0, 1,3);
                    Vector3f glob_normal = global_normals.block(vertex,0, 1,3);
                    Vector3f glob_prevVertex = global_prevVertices.block(vertex, 0, 1, 3);
                    Vector3f glob_prevNormal = global_prevNormals.block(vertex,0, 1,3);

                    //calculate A and b from G N and V matrices
                    G = constructGmat(glob_vertex);
                    A = (G.transpose() * glob_prevNormal).transpose();
                    b = glob_prevNormal.transpose() * (glob_prevVertex - glob_vertex);

                    // solve for x param vector using decomposition
                    incrementPose = (A.transpose() * A).inverse() * A.transpose() * b;

                    //convert pose param vector (x) to transform matrix
                    transformIncrement = convertPoseToMatrix(incrementPose);

                    // check validity of estimation

                    if (isValidEstimation) {
                        // update current transform matrix
                        transform = transformIncrement * transform_prev;
                        //update previous vertices and normals
                        global_prevVertices = global_vertices;
                        global_prevNormals = global_normals;
                    }
                }
            }
        }
    }
};