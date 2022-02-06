#include <iostream>

#include "Eigen.h"
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include "declarations.h"

using namespace std;
using namespace Eigen;



__global__ void computeNormals(Vertex* vertices, int* vertex_validity, Normal* normals, int depthWidth, int depthHeight, Matrix4f depthExtrinsics){


    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    Matrix3f rot = depthExtrinsics.block(0,0,3,3);

    unsigned int curr_idx = u + v * depthWidth;


    if(v >= depthHeight - 1 || u >= depthWidth - 1)
        return;

    
    Vector3f v1, v2;
    Vector3f vk = Vector3f(vertices[curr_idx].pos[0], vertices[curr_idx].pos[1], vertices[curr_idx].pos[2]);
    Vector3f vk_right = Vector3f(vertices[curr_idx+1].pos[0], vertices[curr_idx+1].pos[1], vertices[curr_idx+1].pos[2]);
    Vector3f vk_up = Vector3f(vertices[curr_idx+depthWidth].pos[0], vertices[curr_idx+depthWidth].pos[1], vertices[curr_idx+depthWidth].pos[2]);

    v1 = vk_right - vk;
    v2 = vk_up - vk;

    normals[curr_idx].val = v1.cross(v2);
    normals[curr_idx].val = rot * normals[curr_idx].val;

    normals[curr_idx].val.normalize();
    
}