#include <iostream>

#include "eigen.h"


class VolumetricFusion{

    float ***F;
    float ***W;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float truncation;
    float grid_size;

    public:

    VolumetricFusion(int grid_size, float min_x, float max_x, float min_y, float max_y, float truncation){
        this->grid_size= grid_size;
        this->truncation = truncation
        this->min_x = min_x;
        this->max_x = max_x;
        this->min_y = min_y;
        this->max_y = max_y;

        for (int i = 0; int i < grid_size; ++i) {
            F[i] = new float*[grid_size];
            W[i] = new float*[grid_size];
            for (int j = 0; int j < grid_size; ++j){
                F[i][j] = new float[grid_size];
                W[i][j] = new float[grid_size];

                for(int k; int k < grid_size; k++){
                    F[i][j][k] = truncation;
                    W[i][j][k] = 0;
                }
            }
        }

    }
}