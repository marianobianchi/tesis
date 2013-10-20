cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(openni_viewer_simple)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (openni_viewer_simple openni_viewer_simple.cpp)
target_link_libraries (openni_viewer_simple ${PCL_LIBRARIES})
