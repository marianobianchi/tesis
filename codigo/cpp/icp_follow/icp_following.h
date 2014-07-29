#ifndef __ICP_FOLLOWING__
#define __ICP_FOLLOWING__


/*
 * Includes para el codigo
 * */

#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>


/*
 * Includes para exportar a python
 * */
#include <boost/python.hpp>


struct ICPResult {
    bool has_converged; // was found by icp?
    float score;        // icp score
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
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


ICPResult icp(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
              pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
              ICPDefaults &icp_defaults);


pcl::PointCloud<pcl::PointXYZ>::Ptr read_pcd(std::string pcd_filename);
void save_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr, std::string fname);


void filter_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  const std::string & field_name,
                  const float & lower_limit,
                  const float & upper_limit);


int points(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
pcl::PointXYZ get_point(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int i);


#endif //__ICP_FOLLOWING__
