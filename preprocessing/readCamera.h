//
// Created by ibrahimov on 25.12.21.
//

#ifndef KINECTFUSION_READCAMERA_H
#define KINECTFUSION_READCAMERA_H
#endif //KINECTFUSION_READCAMERA_H

#include "VirtualSensor.h"




class DataHandler {
public:
    DataHandler() {
        std::string filenameIn = "../Data/rgbd_dataset_freiburg1_xyz/";
        VirtualSensor sensor;
        initVirtualSensor(sensor, filenameIn);
    }

    void initVirtualSensor(VirtualSensor& sensor, std::string& filenameIn) {
        std::cout << "Initializing the virtual sensor.." << std::endl;
        if (!sensor.init(filenameIn)) {
            std::cout << "Failed to initialize the sensor!\n Check the file path again.." << std::endl;
            return;https://www.pornhub.com/view_video.php?viewkey=ph5d39a4ad4b581
        }

    }
};