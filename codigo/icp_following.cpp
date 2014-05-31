#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/registration/icp.h>

#include "seguimiento_common/tipos_basicos.h"
#include "seguimiento_common/rgbd.h"
#include "icp_following.h"

void follow ()
{
    // Datos sacados del archivo desk_1.mat para el frame 5
    int im_c_left = 1;
    int im_c_right = 144;
    int im_r_top = 201;
    int im_r_bottom = 318;
    
    
    /**
     * Dada una imagen en profundidad y la ubicación de un objeto
     * en coordenadas sobre la imagen en profundidad, obtengo la nube
     * de puntos correspondiente a esas coordenadas y me quedo
     * unicamente con los valores máximos y mínimos de las coordenadas
     * "x" e "y" de dicha nube
     * 
     * REVISAR: creo que para PCL "x" corresponde a las columnas e "y" a las filas
     **/
     
    DoubleFloatPair rows_cols_limits = from_flat_to_cloud_limits(
        IntPair(im_r_top, im_c_left), //topleft,
        IntPair(im_r_bottom, im_c_right), //bottomright,
        "videos/rgbd/scenes/desk/desk_1/desk_1_5_depth.png"// depth_filename
    );    
    
    /**
     * Levanto las nubes de puntos
     **/
    
    float r_top_limit = rows_cols_limits.first.first;
    float r_bottom_limit = rows_cols_limits.first.second;
    float c_left_limit = rows_cols_limits.second.first;
    float c_right_limit = rows_cols_limits.second.second;
     
    std::string cloud_in_filename  = "videos/rgbd/scenes/desk/desk_1/desk_1_5.pcd";
    std::string cloud_out_filename = "videos/rgbd/scenes/desk/desk_1/desk_1_7.pcd";
    
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
     * Filtro la nube "fuente" segun los valores obtenidos al inicio,
     * filtro la nube "destino" quedandome solo con puntos (x,y,z)
     * "cercanos" a la ubicacion del objeto en la nube "fuente"
     * 
     * REVISAR: creo que para PCL "x" corresponde a las columnas e "y" a las filas
     **/
    
    // Filter points corresponding to the object being followed
    filter_cloud(         cloud_in, filtered_cloud_in, "y", r_top_limit, r_bottom_limit);
    filter_cloud(filtered_cloud_in, filtered_cloud_in, "x", c_left_limit, c_right_limit);
    //filter_cloud(filtered_cloud_in, filtered_cloud_in, "z", z_lower_limit, z_upper_limit);    
    
    // Define row and column limits for the zone to search the object
    // In this case, we look on a box N times the size of the original
    int N = 4;
    r_top_limit = r_top_limit - ( (r_bottom_limit - r_top_limit) * N);
    r_bottom_limit = r_bottom_limit + ( (r_bottom_limit - r_top_limit) * N);
    c_left_limit = c_left_limit - ( (c_right_limit - c_left_limit) * N);
    c_right_limit = c_right_limit + ( (c_right_limit - c_left_limit) * N);
    
    // Filter points corresponding to the zone where the object being followed is supposed to be
    filter_cloud(         cloud_out, filtered_cloud_out, "y", r_top_limit, r_bottom_limit);
    filter_cloud(filtered_cloud_out, filtered_cloud_out, "x", c_left_limit, c_right_limit);
    //filter_cloud(filtered_cloud_out, filtered_cloud_out, "z", z_lower_limit, z_upper_limit);
    
    /**
     * Calculo ICP
     **/
    // Calculate icp transformation
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(filtered_cloud_in);
    icp.setInputTarget(filtered_cloud_out);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    
    // Show some results from icp
    std::cout << "has converged: " << icp.hasConverged() << std::endl;
    std::cout << "score: " << icp.getFitnessScore() << std::endl;
    
    
    
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
                  
    std::cout << "Top = "    << row_top_limit << std::endl;
    std::cout << "Bottom = " << row_bottom_limit << std::endl;
    std::cout << "Left = "   << col_left_limit << std::endl;
    std::cout << "Right = "  << col_right_limit << std::endl;
}
