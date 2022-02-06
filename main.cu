#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include "VirtualSensor.h"
#include "core/VolumetricFusion.h"
#include "core/raycasting.h"
#include "core/preprocessing.h"
#include "core/declarations.h"
//#include "core/PoseEstimation.h"
#include <vector>

#define PLOT_GRAPHS 1

using namespace std;
using namespace cv;
using namespace cuda;

int grid_size = 256;
float min_x = -1;
float max_x = 3;
float min_y = -1;
float max_y = 3;
float min_z = -1;
float max_z = 3;

float truncation = 1.0f;

float minDepth = 0.1f;
float maxDepth = 3;

float truncate(float val){
    int sgn = (val >= 0) ? 1 : -1;
    if(val >= -truncation && val <= truncation)
        return fmin(1, val / truncation);
    else
        return -100;
}

void fusion_step(float* F, float* W, Matrix4f pose, float* depthMap, int* validity, float depthmapWidth, float depthmapHeight, Matrix3f instrinsics){

    float delta_x = (max_x - min_x) / grid_size;
    float delta_y = (max_y - min_y) / grid_size;
    float delta_z = (max_z - min_z) / grid_size;



    Matrix3f intrinsicsInverse = instrinsics.inverse();


    Matrix4f poseInverse = pose.inverse();

    int num_positives = 0;
    int num_negatives = 0;


    for (unsigned int i = 0; i < grid_size; i++) {
        for (unsigned int j = 0; j < grid_size; j++) {
            for(unsigned int k = 0 ; k < grid_size; k++){


                Vector4f p(min_x + i * delta_x , min_y + j * delta_y, min_z + k * delta_z, 1.0f);
                Vector3f p3f = Vector3f(p(0),p(1),p(2));



                Vector3f CameraLocation = pose.block(0,3,3,1);

                Vector4f x =  poseInverse * p;
                Vector3f x3f = instrinsics * Vector3f(x(0),x(1),x(2));

                Vector3f xdot(floor(x3f[0] / x3f[2]), floor(x3f[1] / x3f[2]), 1);

                int currIdx = xdot[0] + xdot[1] * depthmapWidth;

                if(xdot[0] < 0 || xdot[0] >= depthmapWidth || xdot[1] < 0 || xdot[1] >= depthmapHeight)
                    continue;
            

                float depthVal = depthMap[currIdx]; 


                if(validity[currIdx] == 0)
                    continue;

                int curr_tsdf_idx = i * grid_size * grid_size + j * grid_size + k;


                float lambda = (intrinsicsInverse * xdot).norm();
                float Fnew = depthVal - (1/ lambda) * ((CameraLocation - p3f).norm());

                if(Fnew > 0)
                    num_positives += 1;
                else
                    num_negatives += 1;

                Fnew = truncate(Fnew);
                
                if(Fnew != -100){

                    float Fold = F[curr_tsdf_idx];

                    Vector3f PixelRay = intrinsicsInverse * xdot;
                    float Wnew = 1;

                    float Wold = W[curr_tsdf_idx];
                    
                    F[curr_tsdf_idx] = (Wold * Fold + Wnew * Fnew) / (Wold + Wnew);

                    W[curr_tsdf_idx] = Wold + Wnew;

                }
            }
        }
    }
    cout << num_positives << " " << num_negatives << endl;
}

Matrix4f estimatePose(Vertex* vertices, Normal* normals, Vector3f* predictedVertices, Vector3f* predictedNormals, int* vertex_validity, Matrix4f& previousPose, Matrix3f& intrinsics, float depthWidth, float depthHeight) {

    MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
    MatrixXf A_transpose = Eigen::Matrix<float, 6,6>::Zero();
    MatrixXf b = Eigen::Matrix<float, 1, 1>::Zero();
    MatrixXf ATA = Eigen::Matrix<float, 6,6>::Zero();
    MatrixXf ATb = Eigen::Matrix<float, 6,1>::Zero();
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
                    Ng.val = curr_transform.block(0,0,3,3) * normals[curr_idx].val;

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

                    Matrix3f currRot = curr_transform.block(0,0,3,3);

                    float distance = (curr_transform * vertices[curr_idx].pos - predictedVertices[idx]).norm();
                    float normalSim = ((currRot * normals[curr_idx].val).dot(predictedNormals[idx]));

                    if(isnan(normalSim) || isnan(distance))
                        continue;

                    if(distance >= 0.005 || normalSim < 0.9){
                        continue;
                    }


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

//        cout << "calculated pose" << endl;
//        cout << curr_transform << endl;
        curr_transform = transfer_increment * curr_transform;

        cout << curr_transform << endl;
        
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

//    PoseEstimation poseEstimation;
    Matrix<float,4,4> previousPose;

    int sensor_frame = 0;

    Vector3f* predictedNormals = new Vector3f[640*480];
    Vector3f* predictedVertices = new Vector3f[640*480];
    Matrix4f depthExtrinsics = Matrix4f::Identity();
    Matrix4f depthExtrinsicsInv = Matrix4f::Identity();
    while(sensor.processNextFrame()) {
        sensor_frame+= 1;
        float* depthMat = sensor.getDepth();
        unsigned int depthWidth = sensor.getDepthImageWidth();
        unsigned int depthHeight = sensor.getDepthImageHeight();

        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();     // get K matrix (intrinsics), global to camera frame
        Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();
        Matrix3f intrinsicsInv = depthIntrinsics.inverse();

        Matrix4f initialPose;

        if(sensor_frame == 1) {
            depthExtrinsics = Matrix4f::Identity();
            depthExtrinsicsInv = depthExtrinsics.inverse();
            previousPose = depthExtrinsics;
        }


        cv::Mat depth_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, depthMat);
        cv::Mat filt_depth_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F);

        cv::Mat depth_normals(depth_mat.size(), CV_32FC3);

        //appy bilateralFilter to raw depth
        GpuMat filt_depth_mat_g;
        GpuMat depth_map_g;

        depth_map_g.upload(depth_mat);
        cv::cuda::bilateralFilter(depth_map_g, filt_depth_mat_g, 9, 9, 0, BORDER_DEFAULT); //max d val of 5 recommended for real-time applications (9 for offline)
        filt_depth_mat_g.download(filt_depth_mat);

        //depthMat = (float*) filt_depth_mat.data;
        int numVertices = depthHeight * depthWidth;

        Vertex* vertices = new Vertex[numVertices];
        Normal normals[640*480];
        Vertex* vertices_d;
        Normal* normals_d;

        int* vertex_validity = new int[numVertices];
        int* vertex_validity_d;

        unsigned int vertex_idx = -1;
        for (unsigned int r = 0; r < depthHeight; r++) {
            for (unsigned int c = 0; c < depthWidth; c++) {
                vertex_idx += 1;
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



        cudaMalloc(&normals_d, numVertices*sizeof(Normal));
        cudaMalloc(&vertices_d, numVertices*sizeof(Vertex));
        cudaMalloc(&vertex_validity_d, numVertices*sizeof(int));

        cudaMemcpy(vertices_d, vertices, numVertices*sizeof(Vertex), cudaMemcpyHostToDevice);
        cudaMemcpy(vertex_validity_d, vertex_validity, numVertices*sizeof(int), cudaMemcpyHostToDevice);
        
        dim3 threads(30,30);
        dim3 blocks((depthWidth+29) / 30, (depthHeight+29) / 30);
        computeNormals<<<blocks,threads>>>(vertices_d, vertex_validity_d, normals_d, depthWidth, depthHeight);
        //fusion_step(F, W, depthExtrinsics, depthMat, vertex_validity, depthWidth, depthHeight, depthIntrinsics);
        cudaMemcpy(normals, normals_d, numVertices*sizeof(Normal), cudaMemcpyDeviceToHost);
        cudaFree(vertices_d);
        cudaFree(vertex_validity_d);

        if(sensor_frame != 1){

            depthExtrinsics = estimatePose(vertices, normals, predictedVertices, predictedNormals, vertex_validity,
            depthExtrinsics, depthIntrinsics, depthWidth, depthHeight);

            depthExtrinsicsInv = depthExtrinsics.inverse();

            delete[] predictedNormals;
            delete[] predictedVertices;

            predictedNormals = new Vector3f[640*480];
            predictedVertices = new Vector3f[640*480];

        }



        float* phongSurface = new float[depthWidth*depthHeight];

        Vector3f* predictedNormals_d;
        Vector3f* predictedVertices_d;

        float num_voxels = grid_size*grid_size*grid_size;

        float* phongSurface_d;

        float* F_d;
        float* W_d;
        float* depth_d;

        cudaMalloc(&F_d, num_voxels*sizeof(float));

        cudaMalloc(&W_d, num_voxels*sizeof(float));
        cudaMalloc(&vertex_validity_d, numVertices*sizeof(int));
        cudaMalloc(&depth_d, numVertices*sizeof(float));
        cudaMalloc(&predictedNormals_d, numVertices*sizeof(Vector3f));
        cudaMalloc(&predictedVertices_d, numVertices*sizeof(Vector3f));
        cudaMalloc(&phongSurface_d, numVertices*sizeof(float));
        

        cudaMemcpy(F_d, F, num_voxels*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(W_d, W, num_voxels*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(vertex_validity_d, vertex_validity, numVertices*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(depth_d, depthMat, numVertices*sizeof(float), cudaMemcpyHostToDevice);


        dim3 threads_tsdf(5,5,5);
        dim3 blocks_tsdf((grid_size+4) / 5, (grid_size+4) / 5, (grid_size+4) / 5);

//        updateTSDF<<<blocks_tsdf, threads_tsdf>>>(F_d, W_d, depthExtrinsics, depthExtrinsicsInv, depth_d, normals_d, vertex_validity_d, depthWidth, depthHeight, depthIntrinsics, depthIntrinsicsInv, grid_size, truncation, min_x, max_x, min_y, max_y, min_z, max_z);
        updateTSDF<<<blocks_tsdf, threads_tsdf>>>(F_d, W_d, depthExtrinsics, depthExtrinsicsInv, depth_d, normals_d, vertex_validity_d, depthWidth, depthHeight, depthIntrinsics, depthIntrinsicsInv, grid_size, truncation, min_x, max_x, min_y, max_y, min_z, max_z);
        cudaDeviceSynchronize();


        RenderTSDF<<<blocks,threads>>>(F_d, depthExtrinsics, depthIntrinsics, predictedVertices_d, predictedNormals_d, depthWidth, depthHeight, phongSurface_d, grid_size, minDepth, maxDepth, min_x, max_x, min_y, max_y, min_z, max_z);
        cudaDeviceSynchronize();

        cudaMemcpy(F, F_d, num_voxels*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W, W_d, num_voxels*sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(phongSurface, phongSurface_d, numVertices*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(predictedNormals, predictedNormals_d, numVertices*sizeof(Vector3f), cudaMemcpyDeviceToHost);
        cudaMemcpy(predictedVertices, predictedVertices_d, numVertices*sizeof(Vector3f), cudaMemcpyDeviceToHost);
        cudaFree(phongSurface_d);
        cudaFree(predictedNormals_d);
        cudaFree(normals_d);
        cudaFree(vertex_validity_d);
        cudaFree(depth_d);




//        depthExtrinsics_cam = sensor.getTrajectory();
//        cout << "Camera pose: "<< endl << depthExtrinsics_cam << endl;
//        cout << "ICP pose: " << endl << currPose << endl;


//        Matrix4f currPoseInv = currPose.inverse();
//        poseEstimation.updateParams(depthHeight, depthWidth, sensor_frame, vertex_validity, depthIntrinsics);
//        Matrix<float,4,4> currPose = poseEstimation.estimatePose(vertices, normals, predictedVertices, predictedNormals);
//        previousPose = currPose;


    // Test from class func
//        for (unsigned int u_idx = 0; u_idx < numVertices; u_idx++) {
//            cout << "Pred vert: [" << predictedVertices[u_idx][0] << ", " << predictedVertices[u_idx][1] << ", " << predictedVertices[u_idx][2] << "]" << endl;
//            cout << "Pred norm: [" << predictedNormals[u_idx][0] << ", " << predictedNormals[u_idx][1] << ", " << predictedNormals[u_idx][2] << "]" << endl;
//            cout << endl;
//        }
//        poseEstimation.updateParams(depthHeight, depthWidth, sensor_frame, vertex_validity, depthIntrinsics);
//        Matrix<float,4,4> currPose = poseEstimation.estimatePose(vertices, normals, predictedVertices, predictedNormals);
//        depthExtrinsics = currPose;
//        depthExtrinsicsInv = depthExtrinsics.inverse();



        if (PLOT_GRAPHS){
            float cpp_normals[numVertices][3];
            for (unsigned int i = 0; i < numVertices; i++) {
                cpp_normals[i][0] = normals[i].val[0];
                cpp_normals[i][1] = normals[i].val[1];
                cpp_normals[i][2] = normals[i].val[2];
            }

            cv::Mat normalsMap_Vis = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32FC3, predictedNormals);
            cv::Mat phong_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, phongSurface);

            cv::imshow("Normal Map", normalsMap_Vis);
            cv::imshow("Phongsurface ", phong_mat);
            cv::imshow("FilteredDepthMap ", filt_depth_mat);
            cv::imshow("InitialDepth ", depth_mat);
            waitKey(10);
        }

        cout << "-----------------------" << endl;

//        if (sensor_frame >= 5){
//            break;
//        }

//        delete[] vertices;
//        delete[] predictedNormals;
//        delete[] predictedVertices;
//        delete phongSurface;
    }
    return 0;
}