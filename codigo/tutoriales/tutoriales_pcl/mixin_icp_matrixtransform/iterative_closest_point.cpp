#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include "../read_pcd.h"
#include "../visualize_pcd.h"

int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    
    // Fill in the CloudIn data
    read_pcd("../../videos/rgbd/objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_35.pcd", cloud_in);
    
    // Fill in the CloudOut data
    read_pcd("../../videos/rgbd/objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_39.pcd", cloud_out);
    
    // Calculate icp transformation
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_out);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    
    Eigen::Matrix4f transformation_matrix = icp.getFinalTransformation();
    std::cout << transformation_matrix << std::endl;
    
    show_transformation(cloud_in, transformation_matrix);

 return (0);
}
