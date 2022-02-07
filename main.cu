#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include "VirtualSensor.h"

#include "core/VolumetricFusion.h"
#include "core/raycasting.h"
#include "core/preprocessing.h"
#include "core/declarations.h"
#include "core/frame.h"
#include "core/PoseEstimation.h"

#define CALCULATE_DEPTH_LEVELS 0

#define PLOT_RECONSTRUCTION 1
#define PLOT_DEPTH_LEVELS 0

using namespace std;
using namespace cv;
using namespace cuda;

int grid_size = 256;
float min_x = -2;
float max_x = 2;
float min_y = -2;
float max_y = 2;
float min_z = -2;
float max_z = 2;

float truncation = 0.1f;

float minDepth = 0.1f;
float maxDepth = 3;

Matrix4f estimatePose(Vertex* vertices, Normal* normals, Vector3f* predictedVertices, Vector3f* predictedNormals, int* vertex_validity, Matrix4f& previousPose, Matrix3f& intrinsics, float depthWidth, float depthHeight) {

    MatrixXf curr_transform = previousPose;

    MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
    MatrixXf A_transpose = Eigen::Matrix<float, 6,6>::Zero();
    MatrixXf b = Eigen::Matrix<float, 1, 1>::Zero();
    MatrixXf ATA = Eigen::Matrix<float, 6,6>::Zero();
    MatrixXf ATb = Eigen::Matrix<float, 6,1>::Zero();
    MatrixXf previousPoseInv = previousPose.inverse();
    MatrixXf pose;

    for (int iter = 0; iter < 3; iter++) {


        //loop over pixels
        for (int i = 0; i < depthHeight; i++) {
            for (int j = 0; j < depthWidth; j++) {
                int curr_idx = i * depthWidth + j;
                if (vertex_validity[curr_idx] == 1) {
                    Vertex Vg;
                    Normal Ng;

                    //calculate current normals and vertices
                    Vg.pos = previousPose * vertices[curr_idx].pos;
                    Ng.val = previousPose.block(0,0,3,3) * normals[curr_idx].val;

                    //calculate prev normals and vertices
                    Matrix4f frame_transform = previousPoseInv * curr_transform;
                    Vector4f v_c = frame_transform * Vector4f(vertices[curr_idx].pos[0], vertices[curr_idx].pos[1], vertices[curr_idx].pos[2], 1);
                    Vector3f v_c3d = Vector3f(v_c[0], v_c[1], v_c[2]);

                    Vector3f u_hat = intrinsics * v_c3d;
                    u_hat = Vector3f(u_hat[0] / u_hat[2], u_hat[1]/u_hat[2], 1.0f);

                    int idx = u_hat[0] + u_hat[1] * depthWidth;

                    if(idx <= 0 || idx >= depthWidth*depthHeight)
                        continue;


                    if (predictedVertices[idx][0] == MINF) {
                        continue;
                    }

                    float distance = (curr_transform * vertices[curr_idx].pos - predictedVertices[idx]).norm();

                    if(distance >= 0.001)
                        continue;


                    MatrixXf G = Eigen::Matrix<float, 6,6>::Zero();

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
                    b = predictedNormals[idx].transpose() * (predictedNormals[idx] - Vector3f(Vg.pos[0], Vg.pos[1], Vg.pos[2]));


                    ATA += A_transpose * A;
                    ATb += A_transpose * b;

                }
            }
        }
        // solve for pose vector

        pose = ATA.inverse() * ATb;
        cout << "inverted pose" << endl;
        // convert pose vector to transform matrix
        Matrix4f transfer_increment = Matrix4f::Identity();

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

        cout << "calculated pose" << endl;

        cout << curr_transform << endl;

        curr_transform = transfer_increment * curr_transform;

    }

    return curr_transform;
}

int main()
{
    float* W = new float[grid_size*grid_size*grid_size];
    float* F = new float[grid_size*grid_size*grid_size];

    for(int x = 0; x < grid_size; x++) {
        for(int y = 0; y < grid_size; y++) {
            for(int z = 0; z < grid_size; z++) { // initialize the values to whatever you want the default to be
                F[x * grid_size * grid_size + y * grid_size + z] = 1;
                W[x * grid_size * grid_size + y * grid_size + z] = 0;
            }
        }
    }

    // Make sure this path points to the data folder
    std::string filenameIn = "../Data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameBaseOut = "mesh_";

    // load video
    VirtualSensor sensor;
    if (!sensor.init(filenameIn)){
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    } else {
        std::cout << "File Opened" << std::endl;
    }


    Vertex* vertices_d;
    Normal* normals_d;

    float* phongSurface_d;
    float* phongSurface_curr_d;


    float* F_d;
    float* W_d;
    float* depth_d;

    Vector3f* predictedNormals_d;
    Vector3f* predictedVertices_d;
    Vector3f* predictedNormals_curr_d;


    float num_voxels = grid_size*grid_size*grid_size;

    int numVertices = 640*480;

    int* vertex_validity_d;

    cudaMalloc(&normals_d, numVertices*sizeof(Normal));
    cudaMalloc(&vertices_d, numVertices*sizeof(Vertex));
    cudaMalloc(&vertex_validity_d, numVertices*sizeof(int));


    cudaMalloc(&F_d, num_voxels*sizeof(float));

    cudaMalloc(&W_d, num_voxels*sizeof(float));
    cudaMalloc(&vertex_validity_d, numVertices*sizeof(int));
    cudaMalloc(&depth_d, numVertices*sizeof(float));
    cudaMalloc(&predictedNormals_d, numVertices*sizeof(Vector3f));
    cudaMalloc(&predictedNormals_curr_d, numVertices*sizeof(Vector3f));

    cudaMalloc(&predictedVertices_d, numVertices*sizeof(Vector3f));
    cudaMalloc(&phongSurface_d, numVertices*sizeof(float));
    cudaMalloc(&phongSurface_curr_d, numVertices*sizeof(float));


    cudaMemcpy(F_d, F, num_voxels*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_d, W, num_voxels*sizeof(float), cudaMemcpyHostToDevice);


    Matrix4f previousPose;
    Vector3f* predictedNormals = new Vector3f[numVertices];
    Vector3f* predictedVertices = new Vector3f[numVertices];
    Matrix4f depthExtrinsics = Matrix4f::Identity();
    Matrix4f depthExtrinsicsInv = Matrix4f::Identity();


    Frame frame;
    PoseEstimation cameraPose;

    int sensor_frame = 0;
    while(sensor.processNextFrame()) {
        sensor_frame+= 1;
        float* depthMat = sensor.getDepth();
        unsigned int depthWidth = sensor.getDepthImageWidth();
        unsigned int depthHeight = sensor.getDepthImageHeight();

        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();     // get K matrix (intrinsics), global to camera frame
        Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();
        Matrix3f intrinsicsInv = depthIntrinsics.inverse();
//        Matrix4f depthExtrinsics = sensor.getTrajectory();
//        Matrix4f depthExtrinsicsInv = depthExtrinsics.inverse();
        Matrix4f initialPose;

        if(sensor_frame == 1) {
//            initialPose = depthExtrinsics;
            depthExtrinsics = Matrix4f::Identity();
            depthExtrinsicsInv = depthExtrinsics.inverse();
            previousPose = depthExtrinsics;
        }

        cv::Mat depth_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, depthMat);
        cv::Mat filt_depth_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F);
        frame.applyBilateral(depth_mat, filt_depth_mat);

        Vertex* vertices = new Vertex[numVertices];
        Normal* normals = new Normal[numVertices];
        int* vertex_validity = new int[numVertices];


        if (CALCULATE_DEPTH_LEVELS){
            frame.subSampleDepthLevels(filt_depth_mat);
            frame.computeVerticeLevels_cpu(depthIntrinsicsInv);
            frame.computeNormalLevels_cpu();

            vertices = frame.getVertices(1);  //copy original (level1) vertices back for cuda vars
            normals  = frame.getNormals(1);
            vertex_validity = frame.getVertexValidity(1);
        }
        else {
            //otherwise, just calculate once the old way
            for (unsigned int r = 0; r < depthHeight; r++) {
                for (unsigned int c = 0; c < depthWidth; c++) {
                    unsigned int vertex_idx = r * depthWidth + c;
                    float depth_pixel = filt_depth_mat.at<float>(r, c);
                    normals[vertex_idx].val = Vector3f(1.0f, 1.0f, 1.0f);

                    if (isnan(depth_pixel)) {
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

        dim3 threads(30,30);
        dim3 blocks((depthWidth+29) / 30, (depthHeight+29) / 30);

        cudaMemcpy(vertices_d, vertices, numVertices*sizeof(Vertex), cudaMemcpyHostToDevice);
        cudaMemcpy(vertex_validity_d, vertex_validity, numVertices*sizeof(int), cudaMemcpyHostToDevice);

        computeNormals<<<blocks,threads>>>(vertices_d, vertex_validity_d, normals_d, depthWidth, depthHeight);
        cudaMemcpy(normals, normals_d, numVertices*sizeof(Normal), cudaMemcpyDeviceToHost);

        cudaMemcpy(vertex_validity_d, vertex_validity, numVertices*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(depth_d, (float*) filt_depth_mat.data, numVertices*sizeof(float), cudaMemcpyHostToDevice);



        if (sensor_frame != 1){
            depthExtrinsics = estimatePose(vertices, normals, predictedVertices, predictedNormals, vertex_validity,
                                           depthExtrinsics, depthIntrinsics, depthWidth, depthHeight);
            depthExtrinsicsInv = depthExtrinsics.inverse();

            delete[] predictedNormals;
            delete[] predictedVertices;

            predictedNormals = new Vector3f[640*480];
            predictedVertices = new Vector3f[640*480];
        }

        Vector3f* predictedNormals = new Vector3f[depthWidth*depthHeight];
        Vector3f* predictedVertices = new Vector3f[depthWidth*depthHeight];
        float* phongSurface = new float[depthWidth*depthHeight];
        float* phongSurface_curr = new float[depthWidth*depthHeight];

        dim3 threads_tsdf(10,10,10);
        dim3 blocks_tsdf((grid_size+9) / 10, (grid_size+9) / 10, (grid_size+9) / 10);
        updateTSDF<<<blocks_tsdf, threads_tsdf>>>(F_d, W_d, depthExtrinsics, depthExtrinsicsInv, depth_d, normals_d, vertex_validity_d, depthWidth, depthHeight, depthIntrinsics, depthIntrinsicsInv, grid_size, truncation, min_x, max_x, min_y, max_y, min_z, max_z);
        cudaDeviceSynchronize();

        RenderTSDF<<<blocks,threads>>>(F_d, initialPose, depthIntrinsics,predictedVertices_d, predictedNormals_d, depthWidth, depthHeight, phongSurface_d, grid_size, minDepth, maxDepth, min_x, max_x, min_y, max_y, min_z, max_z);
        cudaDeviceSynchronize();

        RenderTSDF<<<blocks,threads>>>(F_d, depthExtrinsics, depthIntrinsics,predictedVertices_d, predictedNormals_curr_d, depthWidth, depthHeight, phongSurface_curr_d, grid_size, minDepth, maxDepth, min_x, max_x, min_y, max_y, min_z, max_z);
        cudaDeviceSynchronize();

        cudaMemcpy(phongSurface, phongSurface_d, numVertices*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(phongSurface_curr, phongSurface_curr_d, numVertices*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(predictedNormals, predictedNormals_d, numVertices*sizeof(Vector3f), cudaMemcpyDeviceToHost);
        cudaMemcpy(predictedVertices, predictedVertices_d, numVertices*sizeof(Vector3f), cudaMemcpyDeviceToHost);



        //-------------------------- plot results ------------------------
        if (PLOT_RECONSTRUCTION) {
            float cpp_normals[numVertices][3];
            for (unsigned int i = 0; i < numVertices; i++) {
                cpp_normals[i][0] = normals[i].val[0];
                cpp_normals[i][1] = normals[i].val[1];
                cpp_normals[i][2] = normals[i].val[2];
            }

            cv::Mat normalsMap_Vis = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32FC3, predictedNormals);
            cv::Mat curr_normals = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32FC3, cpp_normals);
            cv::Mat phong_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, phongSurface);
            cv::Mat phong_mat_curr = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, phongSurface_curr);

            cv::imshow("Normal Map", normalsMap_Vis);
            cv::imshow("Curr normals", curr_normals);
            cv::imshow("Phong Surface ", phong_mat);
            cv::imshow("Phong Surface curr ", phong_mat_curr);

            cv::imshow("Filtered Depth Map ", filt_depth_mat);
            cv::imshow("Initial Depth ", depth_mat);
            waitKey(10);
        }
        if (PLOT_DEPTH_LEVELS){
            frame.plotDepthAndNormals();
        }
    }

    return 0;
}