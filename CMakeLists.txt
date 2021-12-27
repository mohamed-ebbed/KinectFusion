cmake_minimum_required(VERSION 3.11)
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE -Who-dev)

project(KinectFusion)
add_compile_options(-std=c++14 -Wall)

## Define Library directories
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Libs CACHE PATH "Path to lib folder")
set(OpenCV_DIR ${LIBRARY_DIR}/OpenCV/lib/cmake/opencv4 CACHE PATH "Path to installed Opencv")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake/ CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/glog-lib/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/lib/cmake/Ceres CACHE PATH "Path to installed Ceres")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")

find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

#get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
#get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

#add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

include_directories(preprocessing)
set(SOURCE_FILES
    preprocessing/FreeImageHelper.cpp
)
set(HEADER_FILES
    preprocessing/VirtualSensor.h
    preprocessing/Eigen.h
    preprocessing/NearestNeighbor.h
    preprocessing/readCamera.h
    preprocessing/FreeImageHelper.h
)

add_executable(KinectFusion main.cpp ${HEADER_FILES} ${SOURCE_FILES})

#link_directories(${FreeImage_LIBRARY_DIR})
add_executable(main main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(main ceres freeimage Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR})
message(STATUS "Free image include dir:" ${FreeImage_INCLUDE_DIR})
