#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include "VirtualSensor.h"
#include "core/VolumetricFusion.h"
#include "core/declarations.h"
#include <vector>

using namespace std;
using namespace cv;
using namespace cuda;

Matrix4f estimatePose(Vertex* vertices, Normal* normals, Vector3f* predictedVertices, Vector3f* predictedNormals, int* vertex_validity, Matrix4f& previousPose, Matrix3f& intrinsics, float depthWidth, float depthHeight) {

    MatrixXf curr_transform = previousPose;

    for (int iter = 0; iter < 10; iter++) {
        MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
        MatrixXf A_transpose = Eigen::Matrix<float, 6,6>::Zero();
        MatrixXf b = Eigen::Matrix<float, 1, 1>::Zero();
        MatrixXf ATA = Eigen::Matrix<float, 6,6>::Zero();
        MatrixXf ATb = Eigen::Matrix<float, 6,1>::Zero();
        MatrixXf previousPoseInv = previousPose.inverse();
        MatrixXf pose;

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
                    Vector3f u_hat = intrinsics * v_c3d;
                    u_hat = Vector3f(u_hat[0] / u_hat[2], u_hat[1]/u_hat[2], 1.0f);

                    int idx = u_hat[0] + u_hat[1] * depthWidth;

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
                    b = predictedNormals[idx].transpose() * (predictedNormals[idx] - Vector3f(Vg.pos[0], Vg.pos[1], Vg.pos[2]));  // TODO num of prev vertices??

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
        
        curr_transform = pose * curr_transform;

    }

    return curr_transform;
}

__global__ void RenderTSDF(float* tsdf, Matrix4f pose, Matrix3f intrinsics, Vector3f* surfacePoints, Vector3f* predictedNormals, int width, int height, float* phongSurface, float grid_size, float minDepth, float maxDepth, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z) {
    
    int currIdx = -1;

    float delta_x = (max_x - min_x) / grid_size;
    float delta_y = (max_y - min_y) / grid_size;
    float delta_z = (max_z - min_z) / grid_size;

    int num_hits = 0;

    Matrix3f intrinsicsInv = intrinsics.inverse();


    Vector3f CameraLocation = pose.block(0,3,3,1);


    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;   


    if(r >= height || c >= width)
        return;


    float currDepth = minDepth;

    float step = 0.01;

    bool notCrossed = true;

    Vector3f prevPos(0,0,0);

    
    int firstStep = 1;

    float lastVal;


    while(currDepth <= maxDepth && notCrossed){


        currIdx = c + r * width;



        Eigen::Vector3f p_dehom(c , r , 1);
        Eigen::Vector3f p_cam = currDepth * intrinsicsInv * p_dehom;
        Eigen::Vector4f p_cam4f(p_cam[0], p_cam[1], p_cam[2], 1.0);
        Eigen::Vector4f p_world = pose * p_cam4f;
        Eigen::Vector3f pointRay(p_world[0], p_world[1], p_world[2]);



        int i = floor(((pointRay[1] - min_y) / (max_y - min_y)) * (grid_size-1));
        int j = floor(((pointRay[0] - min_x) / (max_x - min_x)) * (grid_size-1));
        int k = floor(((pointRay[2] - min_z) / (max_z - min_z)) * (grid_size-1));




        if(i < 0 || i >= grid_size || j < 0 || j >= grid_size || k < 0 || k >= grid_size)
            break;

        


        int curr_sdf_pos = i * grid_size * grid_size + j * grid_size + k;

        int fx_idx = (i+1) * grid_size * grid_size + j * grid_size + k;
        int fy_idx = (i) * grid_size * grid_size + (j+1) * grid_size + k;
        int fz_idx = (i) * grid_size * grid_size + j * grid_size + (k+1);

        int max_size= grid_size*grid_size*grid_size;

        if(fx_idx < 0 || fx_idx >= max_size || fy_idx < 0 || fy_idx >= max_size || fz_idx < 0 || fz_idx >= max_size)
            break;
        
        


        float currVal = tsdf[curr_sdf_pos];




        
        int currsgn = (currVal >= 0) ? 1 : -1;

        if(firstStep){
            lastVal = currVal;
            firstStep = 0;
            currDepth += step;
            continue;
        }

        int prevsign = (lastVal >= 0) ? 1 : -1;

        if(currsgn != prevsign){
            notCrossed = false;
        }




        if(currVal == 0){
            surfacePoints[currIdx] = pointRay;
            num_hits += 1;


            float fx = (tsdf[fx_idx] - currVal) / (delta_x);
            float fy = (tsdf[fy_idx] - currVal) / (delta_y);
            float fz = (tsdf[fz_idx] - currVal) / (delta_z);

            Vector3f normal(fx,fy,fz);
            normal = normal / normal.norm();
            Vector3f pointToCamera = CameraLocation - pointRay;
    

            pointToCamera.normalize();

            float light = pointToCamera.dot(normal);

            Vector3f delta = pointRay - prevPos;
            Vector3f surface = pointRay - delta * (lastVal / (currVal + lastVal));
            surfacePoints[currIdx] = surface;

            predictedNormals[currIdx] = normal;
            phongSurface[currIdx] = light;
            break;
        } 
        else if(prevsign != currsgn){

            Vector3f delta = pointRay - prevPos;
            Vector3f surface = pointRay - delta * (lastVal / (currVal + lastVal));
            surfacePoints[currIdx] = surface;

            float fx = (tsdf[fx_idx] - currVal) / (delta_x);
            float fy = (tsdf[fy_idx] - currVal) / (delta_y);
            float fz = (tsdf[fz_idx] - currVal) / (delta_z);

            Vector3f normal(fx,fy,fz);
            normal = normal / normal.norm();

            Vector3f pointToCamera = CameraLocation - pointRay;

            pointToCamera.normalize();

            float light = pointToCamera.dot(normal);

            predictedNormals[currIdx] = normal;
            phongSurface[currIdx] = light;                
            num_hits += 1;
    
            break;
        }
        currDepth += step;
        lastVal = currVal;
        prevPos = pointRay;
    }
}


__global__ void updateTSDF(float* F, float* W, float* F_out, float* W_out, Matrix4f pose, float* depthMap, int* validity, float depthmapWidth, float depthmapHeight, Matrix3f instrinsics, int grid_size, int truncation, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z){
    
    float delta_x = (max_x - min_x) / grid_size;
    float delta_y = (max_y - min_y) / grid_size;
    float delta_z = (max_z - min_z) / grid_size;


    Matrix3f intrinsicsInverse = instrinsics.inverse();

    Matrix4f poseInverse = pose.inverse();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;


    if(i >= grid_size || j >= grid_size || k >= grid_size)
        return;

    int curr_sdf_pos = i * grid_size * grid_size + j * grid_size + k;


    F_out[curr_sdf_pos] = 1;


    Vector4f p(min_x + j * delta_x , min_y + i * delta_y, min_z + k * delta_z, 1.0f);
    Vector3f p3f = Vector3f(p(0),p(1),p(2));

    curr_sdf_pos = i * grid_size * grid_size + j * grid_size + k;

    Vector3f CameraLocation = pose.block(0,3,3,1);

    Vector4f x =  poseInverse * p;
    Vector3f x3f = instrinsics * Vector3f(x(0),x(1),x(2));

    Vector3f xdot(floor(x3f[0] / x3f[2]), floor(x3f[1] / x3f[2]), 1);

    int currIdx = xdot[0] + xdot[1] * depthmapWidth;

    if(isnan(xdot[0]) || isnan(xdot[1]))
        return;

    if(xdot[0] < 0 || xdot[0] >= depthmapWidth || xdot[1] < 0 || xdot[1] >= depthmapHeight)
        return;



    float depthVal = depthMap[currIdx]; 

    if(validity[currIdx] == 0)
        return;



    float lambda = (intrinsicsInverse * xdot).norm();
    float Fnew = depthVal - (1/ lambda) * ((CameraLocation - p3f).norm());

    int sgn = (Fnew >= 0) ? 1 : -1;

    if(Fnew >= -truncation && Fnew <= truncation){
        if(1 < Fnew / truncation){
            Fnew = 1;
        }
        else {
            Fnew = Fnew / truncation;
        }
    }
    else{
        Fnew = sgn*truncation;
    }

    
    
    if(Fnew != truncation*sgn){

        float Fold = F[curr_sdf_pos];


        float Wold = W[curr_sdf_pos];

        float Wnew = 1;

        
        F_out[curr_sdf_pos] = (Wold * Fold + Wnew * Fnew) / (Wold + Wnew);

        W_out[curr_sdf_pos] = Wold + Wnew;

    }
}

int main()
{
    int grid_size = 256;
    float min_x = -1.5;
    float max_x = 1.5;
    float min_y = -1.5;
    float max_y = 1.5;
    float min_z = -1.5;
    float max_z = 1.5;

    float truncation = 5.0f;

    float minDepth = 0.01f;
    float maxDepth = 4.0;


    float* W = new float[grid_size*grid_size*grid_size];
    float* F = new float[grid_size*grid_size*grid_size];

    for(int x = 0; x < grid_size; x++) {
        for(int y = 0; y < grid_size; y++) {
            for(int z = 0; z < grid_size; z++) { // initialize the values to whatever you want the default to be
                F[x * grid_size * grid_size + y * grid_size + z] = truncation;
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

    int sensor_frame = 0;

    Matrix4f depthExtrinsics = Matrix4f::Identity();
    Matrix4f depthExtrinsicsInv = depthExtrinsics;


    Matrix4f initialPose = depthExtrinsics;
    while(sensor.processNextFrame()) {
        sensor_frame+= 1;
        float* depthMat = sensor.getDepth();
        unsigned int depthWidth = sensor.getDepthImageWidth();
        unsigned int depthHeight = sensor.getDepthImageHeight();

        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();     // get K matrix (intrinsics), global to camera frame
        Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();
        Matrix3f intrinsicsInv = depthIntrinsics.inverse();


        cv::Mat depth_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, depthMat);
        cv::Mat filt_depth_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F);

        cv::Mat depth_normals(depth_mat.size(), CV_32FC3);

        // //appy bilateralFilter to raw depth

        GpuMat filt_depth_mat_g;
        GpuMat depth_map_g;

        depth_map_g.upload(depth_mat);
        cv::cuda::bilateralFilter(depth_map_g, filt_depth_mat_g, 9, 75, 75, BORDER_DEFAULT); //max d val of 5 recommended for real-time applications (9 for offline)
        filt_depth_mat_g.download(filt_depth_mat);

        depthMat = (float*) filt_depth_mat.data;

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
                normals[vertex_idx].val = Vector3f(0.0f, 0.0f, 0.0f);

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

        cudaMemcpy(normals, normals_d, numVertices*sizeof(Normal), cudaMemcpyDeviceToHost);
    
        cudaFree(normals_d);
        cudaFree(vertices_d);
        cudaFree(vertex_validity_d);


        float cpp_normals[numVertices][3];

        Vector3f* predictedNormals = new Vector3f[depthWidth*depthHeight];
        Vector3f* predictedVertices = new Vector3f[depthWidth*depthHeight];
        float* phongSurface = new float[depthWidth*depthHeight];

        Vector3f* predictedNormals_d;
        Vector3f* predictedVertices_d ;
        float* phongSurface_d;

        float* F_in_d;
        float* W_in_d;
        float* F_out_d;
        float* W_out_d;
        float* depth_d;

        cudaMalloc(&F_in_d, grid_size*grid_size*grid_size*sizeof(float));
        cudaMalloc(&F_out_d, grid_size*grid_size*grid_size*sizeof(float));
        cudaMalloc(&W_in_d, grid_size*grid_size*grid_size*sizeof(float));
        cudaMalloc(&W_out_d, grid_size*grid_size*grid_size*sizeof(float));
        cudaMalloc(&vertex_validity_d, numVertices*sizeof(int));
        cudaMalloc(&depth_d, numVertices*sizeof(float));
        cudaMalloc(&predictedNormals_d, numVertices*sizeof(Vector3f));
        cudaMalloc(&predictedVertices_d, numVertices*sizeof(Vector3f));
        cudaMalloc(&phongSurface_d, numVertices*sizeof(float));
        

        cudaMemcpy(F_in_d, F, grid_size*grid_size*grid_size*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(W_in_d, W, grid_size*grid_size*grid_size*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(vertex_validity_d, vertex_validity, numVertices*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(depth_d, depthMat, numVertices*sizeof(float), cudaMemcpyHostToDevice);


        dim3 threads_tsdf(10,10,10);
        dim3 blocks_tsdf((grid_size+9) / 10, (grid_size+9) / 10, (grid_size+9) / 10);

        updateTSDF<<<blocks_tsdf, threads_tsdf>>>(F_in_d, W_in_d, F_out_d, W_out_d, depthExtrinsics, depth_d, vertex_validity_d, depthWidth, depthHeight, depthIntrinsics, grid_size, truncation, min_x, max_x, min_y, max_y, min_z, max_z);
        
        cudaThreadSynchronize();

        cudaMemcpy(F, F_out_d, grid_size*grid_size*grid_size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W, W_out_d, grid_size*grid_size*grid_size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(F_in_d);
        cudaFree(W_in_d);
        cudaFree(vertex_validity_d);
        cudaFree(depth_d);
        

        RenderTSDF<<<blocks,threads>>>(F_out_d, initialPose, depthIntrinsics,predictedVertices_d, predictedNormals_d, depthWidth, depthHeight, phongSurface_d, grid_size, minDepth, maxDepth, min_x, max_x, min_y, max_y, min_z, max_z);
        cudaThreadSynchronize();


        cudaMemcpy(phongSurface, phongSurface_d, numVertices*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(predictedNormals, predictedNormals_d, numVertices*sizeof(Vector3f), cudaMemcpyDeviceToHost);
        cudaMemcpy(predictedVertices, predictedVertices_d, numVertices*sizeof(Vector3f), cudaMemcpyDeviceToHost);
        cudaFree(phongSurface_d);
        cudaFree(predictedNormals_d);
        cudaFree(F_out_d);
        cudaFree(W_out_d);

        for (unsigned int i = 0; i < numVertices; i++) {
            cpp_normals[i][0] = normals[i].val[0];
            cpp_normals[i][1] = normals[i].val[1];
            cpp_normals[i][2] = normals[i].val[2];
        }

        cv::Mat normalsMap_Vis = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32FC3, predictedNormals);
        cv::Mat phong_mat = cv::Mat(static_cast<int>(depthHeight), static_cast<int>(depthWidth), CV_32F, phongSurface);


        cv::imshow("Normal Map", normalsMap_Vis);
        cv::imshow("Phongsurface ", phong_mat);

        waitKey(0);
        


    }

    return 0;
}