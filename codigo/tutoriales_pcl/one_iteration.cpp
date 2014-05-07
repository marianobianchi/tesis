#include <iostream>

#include <pcl/registration/icp.h>

#include "read_pcd.h"
#include "mixin_icp_matrixtransform/pcl_transformation.h"
#include "common.h"


int main (int argc, char** argv)
{
    
    /**
     * Primero seteo valores fijos sacados del ground truth
     * 
     * REVISAR: que valor setear como "depth"
     **/
    // Datos sacados del archivo desk_1.mat
    std::string cloud_in_filename  = "../videos/rgbd/scenes/desk/desk_1/desk_1_5.pcd";
    std::string cloud_out_filename = "../videos/rgbd/scenes/desk/desk_1/desk_1_6.pcd";
    int im_y_left = 2;
    int im_y_right = 145;
    int im_x_top = 202;
    int im_x_bottom = 319;
    
    // TODO: debería buscarlo en el archivo ../../videos/rgbd/scenes/desk/desk_1/desk_1_5_depth.png
    float depth = 0.4858; // promedio de la profundidad
    
    std::pair<float,float> cloudXY_topleft_corner = from_flat_to_cloud(im_x_top, im_y_left, depth);
    std::pair<float,float> cloudXY_bottomright_corner = from_flat_to_cloud(im_x_bottom, im_y_right, depth);
    
    float x_lower_limit;
    float x_upper_limit;
    float y_lower_limit;
    float y_upper_limit;
    
    
    /**
     * Levanto las nubes de puntos
     **/
    
    // Source clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    
    // Target clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    
    // Fill in the CloudIn data
    read_pcd(cloud_in_filename, cloud_in);
    
    // Fill in the CloudOut data
    read_pcd(cloud_out_filename, cloud_out);
    
    /**
     * Filtros
     **/
    
    // Define  x and y limits around the object
    x_lower_limit = cloudXY_topleft_corner.first;
    x_upper_limit = cloudXY_bottomright_corner.first;
    y_lower_limit = cloudXY_topleft_corner.second;
    y_upper_limit = cloudXY_bottomright_corner.second;
    
    // Filter points corresponding to the object being followed
    filter_cloud(         cloud_in, filtered_cloud_in, "x", x_lower_limit, x_upper_limit);
    filter_cloud(filtered_cloud_in, filtered_cloud_in, "y", y_lower_limit, y_upper_limit);
    //filter_cloud(filtered_cloud_in, filtered_cloud_in, "z", z_lower_limit, z_upper_limit);
    
    
    // Define  x and y limits for the zone to search the object
    // In this case, we look on a box 4 times the size of the original
    x_lower_limit = cloudXY_topleft_corner.first - ( (x_upper_limit - x_lower_limit) * 4);
    x_upper_limit = cloudXY_bottomright_corner.first + ( (x_upper_limit - x_lower_limit) * 4);
    y_lower_limit = cloudXY_topleft_corner.second - ( (y_upper_limit - y_lower_limit) * 4);
    y_upper_limit = cloudXY_bottomright_corner.second + ( (y_upper_limit - y_lower_limit) * 4);
    
    // Filter points corresponding to the zone where the object being followed is supposed to be
    filter_cloud(         cloud_out, filtered_cloud_out, "x", x_lower_limit, x_upper_limit);
    filter_cloud(filtered_cloud_out, filtered_cloud_out, "y", y_lower_limit, y_upper_limit);
    //filter_cloud(filtered_cloud_out, filtered_cloud_out, "z", z_lower_limit, z_upper_limit);
    
    /**
     * ICP
     **/
    // Calculate icp transformation
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(filtered_cloud_in);
    icp.setInputTarget(filtered_cloud_out);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    
    // Show some results from icp
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    
    Eigen::Matrix4f transformation_matrix = icp.getFinalTransformation();
    std::cout << transformation_matrix << std::endl;
    
    show_transformation(filtered_cloud_in, transformation_matrix);

    return 0;
}
