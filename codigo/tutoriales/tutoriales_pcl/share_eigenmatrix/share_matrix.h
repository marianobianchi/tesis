
/*
 * Includes para el codigo
 * */ 
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


typedef Eigen::Matrix4f Mat;
typedef std::vector<float> FloatVector;
typedef std::vector<FloatVector > VectorMat;
typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;

VectorMat mat_to_vector(const Mat&);
Mat vector_to_mat(const VectorMat&);

VectorMat get_transformation();
void transform(VectorMat m);
