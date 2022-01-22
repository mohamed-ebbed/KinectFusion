//
// Created by Burkay SABIRSIZ on 09.01.22.
//

#ifndef SURFACE_RAYCASTING_H
#define SURFACE_RAYCASTING_H

#include <iostream>
#include "Eigen.h"

class Raycasting {
    float minDepth;
    float maxDepth;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
    float truncation;
    float grid_size;

public:
    Raycasting(float minDepth, float maxDepth, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z, float truncation, float grid_size);
    Vector3f mapToWorld(float x, float y, float depth, Matrix3f intrinsics, Matrix4f extrinsics);
    void ProcessSDF(float*** tsdf, Matrix4f pose, Matrix3f intrinsics, Vector3f* surfacePoints, Vector3f* predictedNormals, int width, int height);
};


#endif //SURFACE_RAYCASTING_H
