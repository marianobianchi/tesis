
#include <iostream>

#include <pcl/console/parse.h>
#include <pcl/common/time.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>


/*
 *  Define types
 * */
typedef pcl::PointNormal PointN;
typedef pcl::PointXYZ Point3D;

typedef pcl::FPFHSignature33 FeatureT;

typedef pcl::PointCloud<Point3D> PointCloud3D;
typedef pcl::PointCloud<PointN> PointCloudN;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

typedef pcl::search::KdTree<pcl::PointXYZ> KdTree3D;

typedef pcl::FPFHEstimationOMP<PointN,PointN,FeatureT> FeatureEstimationT;

