cmake_minimum_required(VERSION 3.10)
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE -Who-dev)

project(KinectFusion2)

set(CMAKE_CXX_STANDARD 14)

## Define Library directories

set(Eigen3_DIR /usr/local/share/eigen3/cmake "Path to installed Eigen")
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(OpenCV_DIR ${LIBRARY_DIR}/opencv4/lib/cmake/opencv4 CACHE PATH "Path to installed Opencv")

find_package(Eigen3 REQUIRED)
find_package(OpenCV)

include_directories(preprocessing)
set(SOURCE_FILES
    preprocessing/FreeImageHelper.cpp
    core/RayCastingSimple.cpp
)

set(HEADER_FILES
    preprocessing/VirtualSensor.h
    preprocessing/Eigen.h
    preprocessing/NearestNeighbor.h
    preprocessing/FreeImageHelper.h
    core/PoseEstimation.h
    core/VolumetricFusion.h
    core/RaycastingSimple.h
    core/declarations.h
)

add_executable(KinectFusion2 main.cpp ${HEADER_FILES} ${SOURCE_FILES})
link_directories(${FreeImage_LIBRARY_DIR})
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR})
target_link_libraries(KinectFusion2 freeimage Eigen3::Eigen ${OpenCV_LIBS})

message(STATUS "EIGEN INCLUDE DIR:" ${EIGEN3_INCLUDE_DIR})
