#ifndef __CPP_COMMON__
#define __CPP_COMMON__

#include <vector>

#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/*
 *  Define types
 * */
typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;
typedef pcl::visualization::PointCloudColorHandlerCustom<Point3D> ColorHandler3D;
typedef Eigen::Matrix4f Mat;
typedef std::vector<float> FloatVector;
typedef std::vector<FloatVector > VectorMat;


PointCloud3D::Ptr voxel_grid_downsample(PointCloud3D::Ptr cloud, float leaf);

PointCloud3D::Ptr filter_object_from_scene_cloud(PointCloud3D::Ptr object_cloud,
                                                 PointCloud3D::Ptr scene_cloud,
                                                 float radius,
                                                 bool show_values);

PointCloud3D::Ptr read_pcd(std::string pcd_filename);
void save_pcd(PointCloud3D::Ptr, std::string fname);


PointCloud3D::Ptr filter_cloud(PointCloud3D::Ptr cloud,
                  const std::string & field_name,
                  const float & lower_limit,
                  const float & upper_limit);


VectorMat mat_to_vector(const Mat&);
Mat vector_to_mat(const VectorMat&);
PointCloud3D::Ptr transform_cloud(PointCloud3D::Ptr cloud, const VectorMat& transformation);


int points(PointCloud3D::Ptr cloud);
Point3D get_point(PointCloud3D::Ptr cloud, int i);
void show_clouds(std::string title, PointCloud3D::Ptr first_cloud, PointCloud3D::Ptr second_cloud);


struct MinMax3D {
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
};

MinMax3D get_min_max3D(PointCloud3D::Ptr cloud);

Point3D compute_centroid(float x1, float y1, float z1, float x2, float y2, float z2);



#endif //__CPP_COMMON__
