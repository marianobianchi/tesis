#include "common.h"



PointCloud3D::Ptr voxel_grid_downsample(PointCloud3D::Ptr cloud, float leaf){

    PointCloud3D::Ptr downsampled_cloud(new PointCloud3D);
    
    pcl::VoxelGrid<Point3D> grid;
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (cloud);
    grid.filter (*downsampled_cloud);
    
    return downsampled_cloud;

}
