#ifndef __ICP_FOLLOWING__
#define __ICP_FOLLOWING__


/*
 * Includes para el codigo
 * */

#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>


/*
 * Includes para exportar a python
 * */
#include <boost/python.hpp>


struct ICPResult {
    bool has_converged; // was found by icp?
    float score;        // icp score
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud; //point cloud
};


ICPResult icp(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
              pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud);


pcl::PointCloud<pcl::PointXYZ>::Ptr read_pcd(std::string pcd_filename);


void filter_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  const std::string & field_name,
                  const float & lower_limit,
                  const float & upper_limit);


int points(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
pcl::PointXYZ get_point(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int i);


#endif //__ICP_FOLLOWING__
