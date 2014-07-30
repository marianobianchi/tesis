cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(pcl-icp-with-matrix_transform)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (icp_concept icp_concept.cpp)
target_link_libraries (icp_concept ${PCL_LIBRARIES})
