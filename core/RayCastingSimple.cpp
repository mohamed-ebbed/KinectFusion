#include "RaycastingSimple.h"

using namespace std;


Raycasting::Raycasting(float minDepth, float maxDepth, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z, float truncation, float grid_size) {

    this->minDepth = minDepth;
    this->maxDepth = maxDepth;
    this->min_x = min_x;
    this->max_x = max_x;
    this->min_y = min_y;
    this->max_y = max_y;
    this->min_z = min_z;
    this->max_z = max_z;
    this->truncation = truncation;
    this->grid_size=grid_size;
    
}

Vector3f Raycasting::mapToWorld(float x, float y, float depth, Matrix3f intrinsics, Matrix4f extrinsics) {
    Eigen::Vector3f p_dehom(x , y , 1);
    Eigen::Vector3f p_cam = depth * intrinsics.inverse() * p_dehom;
    Eigen::Vector4f p_cam4f(p_cam[0], p_cam[1], p_cam[2], 1.0);
    Eigen::Vector4f p_world = extrinsics.inverse() * p_cam4f;
    Eigen::Vector3f result(p_world[0], p_world[1], p_world[2]);
    return result;
}


void Raycasting::ProcessSDF(float*** tsdf, Matrix4f pose, Matrix3f intrinsics, Vector3f* surfacePoints, Vector3f* predictedNormals, int width, int height) {
    int currIdx = -1;
    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
        
        cout << r << " " << c << endl;
            
        float currDepth = minDepth;

        float step = 0.1;

        float lastVal = 1;

        bool notCrossed = true;

        Vector3f prevPos(0,0,0);

        while(currDepth <= maxDepth && notCrossed){

            currIdx += 1;


            Vector3f pointRay = mapToWorld(c,r,currDepth,intrinsics,pose);


            int i = floor(((pointRay[0] - min_x) / (max_x - min_x)) * (grid_size-1));
            int j = floor(((pointRay[1] - min_y) / (max_y - min_y)) * (grid_size-1));
            int k = floor(((pointRay[2] - min_z) / (max_z - min_z)) * (grid_size-1));

            float currVal = tsdf[i][j][k];

            int currsgn = (currVal >= 0) ? 1 : -1;
            int prevsign = (lastVal >= 0) ? 1 : -1;

            if(currsgn != prevsign){
                notCrossed = false;
            }


            if(currVal == 0){
                surfacePoints[currIdx] = pointRay;
                break;
            } else if(prevsign != currsgn){
                Vector3f delta = pointRay - prevPos;
                Vector3f surface = pointRay - delta * (lastVal / (currVal + lastVal));
                surfacePoints[currIdx] = surface;
                break;
            }
            currDepth += step;
            lastVal = currVal;
            prevPos = pointRay;
        }
        }
    }
}