cmake_minimum_required(VERSION 3.5)
project(weighted_median_filter)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-O0 -std=gnu++11 -fopenmp -fexceptions")

include_directories(
  include
)

add_executable(weighted_median_filter 
                src/main.cpp
                src/ParameterManager.cpp
              )
              
find_package(PkgConfig REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)

pkg_check_modules(OpenCV REQUIRED opencv)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(weighted_median_filter ${OpenCV_LDFLAGS})

include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(weighted_median_filter ${PCL_LIBRARIES})
include_directories(${VTK_INCLUDE_DIRS})
target_link_libraries(weighted_median_filter ${VTK_LIBRARIES})

