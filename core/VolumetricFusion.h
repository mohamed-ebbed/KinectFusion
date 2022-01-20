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

    float F[256][256][256];
    float W[256][256][256];

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

    VolumetricFusion(int grid_size, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z, float truncation, Matrix3f intrinsics, float depthmapWidth, float depthMapHeight){
        cout << "in constructror" << endl;
        this->grid_size= grid_size;
        this->truncation = truncation;
        this->min_x = min_x;
        this->max_x = max_x;
        this->min_y = min_y;
        this->max_y = max_y;
        this->min_z = min_z;
        this->max_z = max_z;
        this->depthmapHeight = depthMapHeight;
        this->depthmapWidth = depthmapWidth;
        this->intrinsics = intrinsics;
        intrinsicsInverse = intrinsics.inverse();

        for (int i = 0;  i < grid_size; i++) {
            for (int j = 0;  j < grid_size; j++){
                for(int k; k < grid_size; k++){
                    F[i][j][k] = 1;
                    W[i][j][k] = 0;
                }
            }
        }
    }

    float truncate(float val){
        int sgn = (val >= 0) ? 1 : -1;
        if(val >= -truncation)
            return fmin(1, abs(val) / truncation) * sgn;
        else
            return MINF;
    }

    void step(Matrix4f pose, float* depthMap, Normal normals[], int* validity){
        cout << "in step " << endl;
        float delta_x = (max_x - min_x) / grid_size;
        float delta_y = (max_y - min_y) / grid_size;
        float delta_z = (max_z - min_z) / grid_size;

        Matrix4f poseInverse = pose.inverse();

        int pixelIdx = -1;
        float f_min = 100.0f;
        float f_max = -100.0f;

        for (unsigned int i = 0; i < 256; i++) {
            for (unsigned int j = 0; j < 256; j++) {
                for(unsigned int k = 0 ; k < 256; k++){
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

                    if(validity[currIdx] == 0) {
                        cout << "F[i][j][k] max" << f_max << endl;
                        cout << "F[i][j][k] min" << f_min << endl;
                        continue;
                    }

                    float lambda = (intrinsicsInverse * xdot).norm();
                    float Fnew = (1/ lambda) * ((CameraLocation - p3f).norm()) - depthVal;
                    
                    if(Fnew != MINF){

                        float Fold = F[i][j][k];

                        Vector3f PixelRay = intrinsicsInverse * xdot;
                        float Wnew = 1;

                        float Wold = W[i][j][k];
                        
                        F[i][j][k] = (Wold * Fold + Wnew * Fnew) / (Wold + Wnew);

                        W[i][j][k] = Wold + Wnew;
//                        cout << "F[i][j][k] = " << F[i][j][k] << endl;


                        if(F[i][j][k] > f_max) {
                            f_max = F[i][j][k] ;
                        }
                        if (F[i][j][k] < f_min) {
                            f_min = F[i][j][k] ;
                        }
                        cout << "F[i][j][k] max" << f_max << endl;
                        cout << "F[i][j][k] min" << f_min << truncate(f_min) << endl;
                        
                    }
                    // cout << "Fnew=" << Fnew << endl;
                }
            }
        }
    }

};