//
// Created by Burkay SABIRSIZ on 09.01.22.
//

#ifndef SURFACE_RAYCASTING_H
#define SURFACE_RAYCASTING_H

#include <iostream>
#include "Eigen.h"

class Raycasting {
public:
    Raycasting();
    void ProcessSDF(float ***tsdf, Matrix4f extrinsics, Matrix4f intrinsics, Matrix4f Tg, MatrixXf depthFrame);
    Vector3f homogenizeToWorldPoint(float x, float y, float depth, Matrix4f intrinsics, Matrix4f extrinsics);
    Vector3f getCameraWorldPoint(Matrix4f extrinsics);
    Vector3f getUnitRay(Vector3f cameraWorldPoint, Vector3f homogenizedWorldPoint);
    Vector3f getSurfacePoints();
};


#endif //SURFACE_RAYCASTING_H
