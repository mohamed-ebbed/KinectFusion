//
// Created by Burkay SABIRSIZ on 09.01.22.
//

#ifndef SURFACE_RAYCASTING_H
#define SURFACE_RAYCASTING_H

#include <iostream>
#include "Eigen.h"

class Raycasting {
public:
    Raycasting(int width, int height);
    void ProcessSDF(float* tsdf, Matrix4f pose, Matrix3f intrinsics);
};


#endif //SURFACE_RAYCASTING_H
