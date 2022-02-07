//
// Created by ibrahimov on 20.01.22.
//

#pragma once

#ifndef KINECTFUSION2_DECLARATIONS_H
#define KINECTFUSION2_DECLARATIONS_H

#endif //KINECTFUSION2_DECLARATIONS_H

#include "Eigen.h"

struct Vertex{
    Vector4f pos;
};

struct Normal {
    Vector3f val;
};


struct Point3D{ // point in the 3D world coordinate
    Vector3f pos;
    Vector3f normal;
};