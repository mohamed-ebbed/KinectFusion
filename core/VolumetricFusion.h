#include <iostream>

#include "Eigen.h"
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include "declarations.h"

using namespace std;
using namespace Eigen;

#define nullval  std::numeric_limits<float>::infinity();


class VolumetricFusion{

    float ***F;
    float ***W;

    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
    float depthmapWidth;
    float depthmapHeight;
    float truncation;
    float grid_size;
    Matrix3f intrinsicsInverse;
    Matrix3f intrinsics;

    public:

    float*** getF(){
        return F;
    }

    VolumetricFusion(int grid_size, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z, float truncation){
        this->grid_size= grid_size;
        this->truncation = truncation;
        this->min_x = min_x;
        this->max_x = max_x;
        this->min_y = min_y;
        this->max_y = max_y;
        this->min_z = min_z;
        this->max_z = max_z;

        W = new float**[grid_size];
        F = new float**[grid_size];

        for(int x = 0; x < grid_size; x++) {
            F[x] = new float*[grid_size];
            W[x] = new float*[grid_size];
            for(int y = 0; y < grid_size; y++) {
                F[x][y] = new float[grid_size];
                W[x][y] = new float[grid_size];
                for(int z = 0; z < grid_size; z++) { // initialize the values to whatever you want the default to be
                    F[x][y][z] = truncation;
                    W[x][y][z] = 0;
                }
            }
        }
        cout << "initalized arrays" << endl;
    }

    float truncate(float val){
        int sgn = (val >= 0) ? 1 : -1;
        if(val >= -truncation)
            return fmin(1, val / truncation) * sgn * sgn;
        else
            return truncation;
    }

    void step(Matrix4f pose, float* depthMap, Normal normals[], int* validity, float depthmapWidth, float depthmapHeight, Matrix3f instrinsics){
        float delta_x = (max_x - min_x) / grid_size;
        float delta_y = (max_y - min_y) / grid_size;
        float delta_z = (max_z - min_z) / grid_size;

        intrinsicsInverse = intrinsics.inverse();


        Matrix4f poseInverse = pose.inverse();

        int pixelIdx = -1;


        for (unsigned int i = 0; i < grid_size; i++) {
            for (unsigned int j = 0; j < grid_size; j++) {
                for(unsigned int k = 0 ; k < grid_size; k++){
                    Vector4f p(min_x + i * delta_x , min_y + j * delta_y, min_z + k * delta_z, 1.0f);
                    Vector3f p3f = Vector3f(p(0),p(1),p(2));

                    Vector3f CameraLocation = pose.block(0,3,3,1);

                    Vector4f x =  poseInverse * p;
                    Vector3f x3f = intrinsics * Vector3f(x(0),x(1),x(2));

                    Vector3f xdot(floor(x3f[0] / x3f[2]), floor(x3f[1] / x3f[2]), 1);

                    if(xdot[0] < 0 || xdot[0] >= depthmapWidth || xdot[1] < 0 || xdot[1] >= depthmapHeight)
                        continue;

                    int currIdx = xdot[0] + xdot[1] * depthmapWidth;
                    float depthVal = depthMap[currIdx]; 

                    float lambda = (intrinsicsInverse * xdot).norm();
                    float Fnew = (1/ lambda) * ((CameraLocation - p3f).norm()) - depthVal;
                    
                    if(Fnew != MINF){

                        float Fold = F[i][j][k];

                        Vector3f PixelRay = intrinsicsInverse * xdot;
                        float Wnew = 1;

                        float Wold = W[i][j][k];
                        
                        F[i][j][k] = (Wold * Fold + Wnew * Fnew) / (Wold + Wnew);

                        W[i][j][k] = Wold + Wnew;


                    }
                }
            }
        }
    }

    ~VolumetricFusion(){
        for(int x = 0; x < grid_size; ++x) {
            for(int y = 0; y < grid_size; ++y) {
                delete [] F[x][y];
                delete [] W[x][y];
            }
            delete [] F[x];
            delete [] W[x];
         }
         delete [] F;
         delete [] W;
    }



};