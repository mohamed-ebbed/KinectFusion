cmake_minimum_required(VERSION 3.11)
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE -Who-dev)

project(KinectFusion2)
add_compile_options(-std=c++14 -Wall)

## Define Library directories
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs "Path to lib folder")
set(OpenCV_DIR /usr/local/lib/cmake/opencv4  "Path to installed Opencv")
#set(glog_DIR ${LIBRARY_DIR}/glog-lib/lib/cmake/glog CACHE PATH "Path to installed glog")
#set(Ceres_DIR ${LIBRARY_DIR}/Ceres/lib/cmake/Ceres CACHE PATH "Path to installed Ceres")
#set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")

#find_package(Ceres REQUIRED)
#find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(EIGEN3_INCLUDE_DIR /Users/mohamedramzy/TUM/First_Semester/3d_scanning/excercises/Libs/Eigen/include/eigen3)
set(FreeImage_INCLUDE_DIR /usr/local/include CACHE PATH "Path to FreeImage header file")
set(FreeImage_LIBRARY_DIR /usr/local/lib CACHE PATH "Path to FreeImage .lib/.dll folder")

#get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
#get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)
add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

include_directories(preprocessing)

set(SOURCE_FILES
    preprocessing/FreeImageHelper.cpp
#    core/Raycasting.cpp
)

set(HEADER_FILES
    preprocessing/VirtualSensor.h
    preprocessing/Eigen.h
    preprocessing/NearestNeighbor.h
    preprocessing/FreeImageHelper.h
    core/PoseEstimation.h
    core/VolumetricFusion.h
#    core/Raycasting.h
    core/declarations.h
)


link_directories(${FreeImage_LIBRARY_DIR})

add_executable(KinectFusion2 main.cpp ${HEADER_FILES} ${SOURCE_FILES})

include_directories(${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(KinectFusion2 general freeimage Eigen3::Eigen ${OpenCV_LIBS})
message(STATUS "Free image include dir:" ${FreeImage_LIBRARY_DIR})



