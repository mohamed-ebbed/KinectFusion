//
// Created by Burkay SABIRSIZ on 09.01.22.
//

#include "Raycasting.h"

Raycasting::Raycasting() {
    
}

//Brings a pixel from current depth camera image to 3D camera world, and then brings that point into
//actual 3D world coordinates by E^-1*K^-1*(dehomegized pixel).
Vector3f Raycasting::homogenizeToWorldPoint(float x, float y, float depth, Matrix4f intrinsics, Matrix4f extrinsics) {
    Eigen::Vector4f homogenizedPixelCoords(x * depth, y * depth, depth, 1.0);
    homogenizedPixelCoords = extrinsics.inverse() * (intrinsics.inverse() * homogenizedPixelCoords);
    Eigen::Vector3f result(homogenizedPixelCoords.x(), homogenizedPixelCoords.y(), homogenizedPixelCoords.z());
    return result;
}

//Get camera location in real world.
Vector3f Raycasting::getCameraWorldPoint(Matrix4f extrinsics) {
    Vector4f col = extrinsics.col(3);
    Vector3f cameraWorldPoint(col.x(), col.y(), col.z());
    return cameraWorldPoint;
}

//Get the unit length ray from camera to given point.
Vector3f Raycasting::getUnitRay(Vector3f cameraWorldPoint, Vector3f homogenizedWorldPoint) {
    Vector3f ray = homogenizedWorldPoint - cameraWorldPoint;
    Vector3f unitRay = ray.normalized();
    return unitRay;
}

void Raycasting::ProcessSDF(float ***tsdf, Matrix4f extrinsics, Matrix4f intrinsics, Matrix4f Tg, MatrixXf depthFrame) {
    Vector3f surfacePoints[depthFrame.rows()][depthFrame.cols()];
    for(int i = 0; i<depthFrame.rows(); i++){
        for(int j = 0; j<depthFrame.cols(); j++){
            Vector3f h_wp = homogenizeToWorldPoint(i,j,depthFrame(i,j),intrinsics,extrinsics);
            Vector3f camera_wp = getCameraWorldPoint(extrinsics);
            Vector3f unitRay = getUnitRay(camera_wp, h_wp);

            Vector3f currentPos(camera_wp.x(), camera_wp.y(), camera_wp.z());
            Vector3f previousPos(camera_wp.x(), camera_wp.y(), camera_wp.z());
            float currentSign = 0;
            bool notCrossed = true;
            float maxDepth = 100;
            float valueAtRc = 1;
            float previousValueAtRc = 1;

            while(notCrossed){
                valueAtRc = tsdf[(int)round(currentPos.x())][(int)round(currentPos.y())][(int)round(currentPos.z())];
                if((valueAtRc == 1 || valueAtRc > currentSign) && currentPos.norm() < maxDepth){
                    currentPos += unitRay;
                    previousValueAtRc = valueAtRc;  
                } else {
                    notCrossed = false;
                }
            }

            if(valueAtRc > 0) {
                std::cout << "couldn't find the surface" << std::endl;
            } else if(valueAtRc == 0){
                surfacePoints[i][j] = currentPos;
            } else if(valueAtRc < 0){
                float multiplier = valueAtRc / (previousValueAtRc - valueAtRc);
                currentPos = currentPos + (unitRay * multiplier);
                surfacePoints[i][j] = currentPos;
            }
        }
    }
}
