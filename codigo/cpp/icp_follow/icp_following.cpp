#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/registration/icp.h>

#include "tipos_basicos.h"
#include "rgbd.h"
#include "icp_following.h"

ICPResult follow (IntPair top_left, IntPair bottom_right, std::string depth_fname, std::string source_cloud_fname, std::string target_cloud_fname)
{
    
    /**
     * Dada una imagen en profundidad y la ubicación de un objeto
     * en coordenadas sobre la imagen en profundidad, obtengo la nube
     * de puntos correspondiente a esas coordenadas y me quedo
     * unicamente con los valores máximos y mínimos de las coordenadas
     * "x" e "y" de dicha nube
     * 
     * REVISAR: creo que para PCL "x" corresponde a las columnas e "y" a las filas
     **/
     
    DoubleFloatPair rows_cols_limits = from_flat_to_cloud_limits(top_left, bottom_right, depth_fname);    
    
    /**
     * Levanto las nubes de puntos
     **/
    
    float r_top_limit = rows_cols_limits.first.first;
    float r_bottom_limit = rows_cols_limits.first.second;
    float c_left_limit = rows_cols_limits.second.first;
    float c_right_limit = rows_cols_limits.second.second;
    
    // Source clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    
    // Target clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    
    // Fill in the CloudIn data
    read_pcd(source_cloud_fname, source_cloud);
    
    // Fill in the CloudOut data
    read_pcd(target_cloud_fname, target_cloud);
    
    /**
     * Filtro la nube "fuente" segun los valores obtenidos al inicio,
     * filtro la nube "destino" quedandome solo con puntos (x,y,z)
     * "cercanos" a la ubicacion del objeto en la nube "fuente"
     * 
     * REVISAR: creo que para PCL "x" corresponde a las columnas e "y" a las filas
     **/
    
    // Filter points corresponding to the object being followed
    filter_cloud(         source_cloud, filtered_source_cloud, "y", r_top_limit, r_bottom_limit);
    filter_cloud(filtered_source_cloud, filtered_source_cloud, "x", c_left_limit, c_right_limit);
    //filter_cloud(filtered_source_cloud, filtered_source_cloud, "z", z_lower_limit, z_upper_limit);    
    
    // Define row and column limits for the zone to search the object
    // In this case, we look on a box N times the size of the original
    int N = 4;
    r_top_limit = r_top_limit - ( (r_bottom_limit - r_top_limit) * N);
    r_bottom_limit = r_bottom_limit + ( (r_bottom_limit - r_top_limit) * N);
    c_left_limit = c_left_limit - ( (c_right_limit - c_left_limit) * N);
    c_right_limit = c_right_limit + ( (c_right_limit - c_left_limit) * N);
    
    // Filter points corresponding to the zone where the object being followed is supposed to be
    filter_cloud(         target_cloud, filtered_target_cloud, "y", r_top_limit, r_bottom_limit);
    filter_cloud(filtered_target_cloud, filtered_target_cloud, "x", c_left_limit, c_right_limit);
    //filter_cloud(filtered_target_cloud, filtered_target_cloud, "z", z_lower_limit, z_upper_limit);
    
    /**
     * Calculo ICP
     **/
    // Calculate icp transformation
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(filtered_source_cloud);
    icp.setInputTarget(filtered_target_cloud);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    
    /**
     * Busco los limites en el dominio de las filas y columnas del RGB
     * */
    int col_left_limit = 639;
    int col_right_limit = 0;
    int row_top_limit = 479;
    int row_bottom_limit = 0;
    
    IntPair flat_xy;
    
    for (int i = 0; i < Final.points.size (); i++){
        flat_xy = from_cloud_to_flat(Final.points[i].y, Final.points[i].x, Final.points[i].z);
        
        if(flat_xy.first < row_top_limit) row_top_limit = flat_xy.first;
        if(flat_xy.first > row_bottom_limit) row_bottom_limit = flat_xy.first;
        
        if(flat_xy.second < col_left_limit) col_left_limit = flat_xy.second;
        if(flat_xy.second > col_right_limit) col_right_limit = flat_xy.second;
    }
    
    ICPResult res;
    res.has_converged = icp.hasConverged();
    res.score = icp.getFitnessScore();
    int width = col_right_limit - col_left_limit;
    int height = row_bottom_limit - row_top_limit;
    res.size = width > height? width: height;
    res.top = row_top_limit;
    res.left = col_left_limit;
    
    return res;
}



/**
 * Exporto todo a python
 * */

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>


void export_follow(){
    
    using namespace boost::python;

    def("follow", follow);
    
}

