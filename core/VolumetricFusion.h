#include <iostream>

#include "Eigen.h"
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include "declarations.h"

using namespace std;
using namespace Eigen;


__global__ void updateTSDF(float* F, float* W, Matrix4f pose, Matrix4f poseInverse, float* depthMap, Normal* normals, int* validity, float depthmapWidth, float depthmapHeight, Matrix3f instrinsics, Matrix3f intrinsicsInverse, int grid_size, float truncation, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z){
    
    float delta_x = (max_x - min_x) / grid_size;
    float delta_y = (max_y - min_y) / grid_size;
    float delta_z = (max_z - min_z) / grid_size;



    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;


    if(i >= grid_size || j >= grid_size || k >= grid_size)
        return;



    Vector4f p(min_x + i * delta_x , min_y + j * delta_y, min_z + k * delta_z, 1.0f);
    Vector3f p3f = Vector3f(p(0),p(1),p(2));



    Vector3f CameraLocation = pose.block(0,3,3,1);

    Vector4f x =  poseInverse * p;
    Vector3f x3f = instrinsics * Vector3f(x(0),x(1),x(2));

    Vector3f xdot(floor(x3f[0] / x3f[2]), floor(x3f[1] / x3f[2]), 1);

    int currIdx = xdot[0] + xdot[1] * depthmapWidth;

    if(xdot[0] < 0 || xdot[0] >= depthmapWidth || xdot[1] < 0 || xdot[1] >= depthmapHeight)
        return;


    float depthVal = depthMap[currIdx]; 


    if(validity[currIdx] == 0)
        return;

    int curr_tsdf_idx = i * grid_size * grid_size + j * grid_size + k;


    float lambda = (intrinsicsInverse * xdot).norm();
    float Fnew = depthVal - (1/ lambda) * ((CameraLocation - p3f).norm());


    if(Fnew >= -truncation && Fnew <= truncation)
        if (1 < Fnew / truncation)
            Fnew = 1;
        else
            Fnew = Fnew / truncation;
    else
        Fnew =  -100;
    
    if(Fnew != -100){

        float Fold = F[curr_tsdf_idx];

        Vector3f PixelRay = intrinsicsInverse * xdot;
        float Wnew = 1;

        float Wold = W[curr_tsdf_idx];
        
        F[curr_tsdf_idx] = (Wold * Fold + Wnew * Fnew) / (Wold + Wnew);

        W[curr_tsdf_idx] = Wold + Wnew;

    }
}
