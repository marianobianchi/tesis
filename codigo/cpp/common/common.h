#ifndef __CPP_COMMON__
#define __CPP_COMMON__

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

/*
 *  Define types
 * */
typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;


PointCloud3D::Ptr voxel_grid_downsample(PointCloud3D::Ptr cloud, float leaf);



#endif //__CPP_COMMON__
