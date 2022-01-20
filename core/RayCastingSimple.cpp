#include "RaycastingSimple.h"


Raycasting::Raycasting(int width, int height, float minDepth, float maxDepth, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z) {

    this->width = width;
    this->height = height;
    this->minDepth = minDepth;
    this->maxDepth = maxDepth;
    this->min_x = min_x;
    this->max_x = max_x;
    this->min_y = min_y;
    this->max_y = max_y;
    this->min_z = min_z;
    this->max_z = max_z;
    
}

Vector3f Raycasting::mapToWorld(float x, float y, float depth, Matrix3f intrinsics, Matrix4f extrinsics) {
    Eigen::Vector3f p_dehom(x , y , 1);
    Eigen::Vector3f p_cam = depth * intrinsics.inverse() * p_dehom;
    Eigen::Vector4f p_cam4f(p_cam[0], p_cam[1], p_cam[2], 1.0);
    Eigen::Vector4f p_world = extrinsics * p_cam4f;
    Eigen::Vector3f result(p_world[0], p_world[1], p_world[2]);
    return result;
}


void Raycasting::ProcessSDF(float*** tsdf, Matrix4f pose, Matrix3f intrinsics, Vector3f* surfacePoints, Vector3f* predictedNormals) {
    int currIdx = -1;
    for(int r = 0; r< height; r++){
        for(int c = 0; width; c++){
            
        float currDepth = minDepth;

        float step = 0.1;

        float lastVal = 1;

        bool notCrossed = true;

        Vector3f prevPos(0,0,0);

        while(currDepth <= maxDepth && notCrossed){

            curr_idx += 1;
            Vector3f pointRay = mapToWorld(c,r,currDepth,intrinsics,extrinsics);

            int i = floor(((pointRay[0] - min_x) / (max_x - min_x)) * 255);
            int j = floor(((pointRay[1] - min_y) / (max_y - min_y)) * 255);
            int k = floor(((pointRay[2] - min_z) / (max_z - min_z)) * 255);

            float currVal = tsdf[i][j][k];

            int currsgn = (currVal >= 0) ? 1 : -1;
            int prevsign = (lastVal >= 0) ? 1 : -1;

            if(currsgn != prevsign){
                notCrossed = false;
            }

            if(currVal > 0) {
                std::cout << "couldn't find the surface" << std::endl;

            } else if(currVal == 0){
                surfacePoints[curr_idx] = pointRay;
                break;
            } else if(prevsign != currsgn){
                Vector3f delta = pointRay - prevPos;
                Vector3f surface = pointRay - delta * (lastVal / (currVal + lastVal));
                surfacePoints[curr_idx] = surface;
                break;
            }
            currDepth += step;
            lastVal = currVal;
            lastPos = pointRay;
        }
        }
    }
}