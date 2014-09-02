#ifndef __ICP_FOLLOWING__
#define __ICP_FOLLOWING__


/*
 * Includes para el codigo
 * */

#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>


/*
 * Define types
 * */
typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;

typedef pcl::visualization::PointCloudColorHandlerCustom<Point3D> ColorHandler3D;

/*
 * Includes para exportar a python
 * */
#include <boost/python.hpp>


struct ICPResult {
    bool has_converged; // was found by icp?
    float score;        // icp score
    pcl::PointCloud<Point3D>::Ptr cloud;
};

struct ICPDefaults {
    ICPDefaults(){
        euc_fit = -1.7976931348623157081e+308;
        max_corr_dist = 1.3407807929942595611e+154;
        max_iter = 10;
        transf_epsilon = 0;
        ran_iter = 0;
        ran_out_rej = 0.05;
        show_values = false;
    }

    double euc_fit;
    double max_corr_dist;
    int max_iter;
    double transf_epsilon;
    int ran_iter;
    double ran_out_rej;
    bool show_values;

};


ICPResult icp(PointCloud3D::Ptr source_cloud,
              PointCloud3D::Ptr target_cloud,
              ICPDefaults &icp_defaults);


PointCloud3D::Ptr read_pcd(std::string pcd_filename);
void save_pcd(PointCloud3D::Ptr, std::string fname);


void filter_cloud(PointCloud3D::Ptr cloud,
                  const std::string & field_name,
                  const float & lower_limit,
                  const float & upper_limit);


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

#endif //__ICP_FOLLOWING__
