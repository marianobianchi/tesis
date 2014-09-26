#ifndef __ALIGNMENT_PREREJECTIVE__
#define __ALIGNMENT_PREREJECTIVE__


/*
 * Includes para el codigo
 * */
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>



/*
 *  Define types
 * */
typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<Point3D> ColorHandler3D;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;




struct APResult {
    bool has_converged; // was found by ap?
    double score;
    PointCloud3D::Ptr cloud;
};

struct APDefaults {
    APDefaults(){
        leaf = 0.004;
        max_ransac_iters = 1000;
        points_to_sample = 5; // needs to be >= 3
        nearest_features_used = 2; // needs to be >= 2
        simil_threshold = 0.7; // bigger is faster but less robust
        inlier_threshold = 2.0; // will be multiplied with leaf
        inlier_fraction = 0.5; // fraction for accepting a pose hypothesis
        show_values = false;
    };

    float leaf;
    int max_ransac_iters;
    int points_to_sample;
    int nearest_features_used;
    float simil_threshold;
    float inlier_threshold;
    float inlier_fraction;
    bool show_values;

};

APResult alignment_prerejective(PointCloud3D::Ptr const_source_cloud,
                                PointCloud3D::Ptr const_target_cloud,
                                APDefaults &ap_defaults);


#endif //__ALIGNMENT_PREREJECTIVE__
