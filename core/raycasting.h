#include <iostream>

#include "Eigen.h"
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include "declarations.h"

using namespace std;
using namespace Eigen;


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



        int i = floor(((pointRay[2] - min_z) / (max_z - min_z)) * (grid_size-1));
        int j = floor(((pointRay[1] - min_y) / (max_y - min_y)) * (grid_size-1));
        int k = floor(((pointRay[0] - min_x) / (max_x - min_x)) * (grid_size-1));



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