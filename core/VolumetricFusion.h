#include <iostream>

#include "eigen.h"

using namespace std;
using namespace Eigen;


class VolumetricFusion{

    float ***F;
    float ***W;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float truncation;
    float grid_size;
    float intrinsicsInverse;

    Matrix3f instrinsics; 

    public:

    VolumetricFusion(int grid_size, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z, float truncation, Matrix3f intrinsics){

        this->grid_size= grid_size;
        this->truncation = truncation
        this->min_x = min_x;
        this->max_x = max_x;
        this->min_y = min_y;
        this->max_y = max_y;
        this->intrinsics = intrinsics;
        this->intrinsicsInverse = intrinsics.inverse();


        this->min_z = min_z;
        this->max_z = max_z;

        for (int i = 0; int i < grid_size; ++i) {
            F[i] = new float*[grid_size];
            W[i] = new float*[grid_size];
            for (int j = 0; int j < grid_size; ++j){
                F[i][j] = new float[grid_size];
                W[i][j] = new float[grid_size];

                for(int k; int k < grid_size; k++){
                    F[i][j][k] = NULL;
                    W[i][j][k] = 0;
                }
            }
        }
    }

    float truncate(float val){
        bool sgn = (val >= 0) ? 1 : -1;

        if(val > truncation)
            return min(1, val / truncation) * sgn;
        else
            return NULL;

    }

    float step(Matrix4f pose, float** depthMap){
        float delta_x = (max_x - min_x) / grid_size;
        float delta_y = (max_y - min_y) / grid_size;
        float delta_z = (max_z - min_z) / grid_size;

        Matrix4f poseInverse = pose.inverse();

        for(int i = 0; i < grid_size; i++){
            for(int j = 0; j < grid_size; j++){
                for(int k = 0; j < grid_size; k++){

                    Vector3f p(min_x + i * delta_x , min_y + j * delta_y, min_z + k * delta_z);

                    Vector3f CameraLocation = pose.block(0,3,3,1);

                    Vector3f x = intrinsics * poseInverse * p;

                    Vector3f xdot(floor(x[0] / x[2]), floor(x[1] / x[2]), 1);

                    float lambda = (intrinsics.inverse() * xdot).norm();

                    float Fnew = truncate((1 / lambda) * (CameraLocation - p).norm() - depthMap[(int)x[0]][(int)x[1]]);
                    float Fold = F[i][j][k]; 

                    Vector3f PixelRay = pose * intrinsicsInverse * xdot - CameraLocation;
                    Vector3f CameraToPoint = (p - CameraLocation)

                    float cosineAngle = (PixelRay * CameraToPoint) / (PixelRay.norm() * CameraToPoint.norm());

                    float Wnew = cosineAngle / depthMap[(int)xdot[0]][(int)xdot[1]];
                    float Wold = W[i][j][k];

                    if(Fold == NULL) Fold = 0;


                    F[i][j][k] = (Wold * Fold + Wnew * Fnew) / (Wold + Wnew);

                    W[i][j][k] = Wold + Wnew;

                }
            }
        }
    }

}