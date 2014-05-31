#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/registration/icp.h>

#include "read_pcd.h"
#include "visualize_pcd.h"
#include "common.h"


int main (int argc, char** argv)
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
    
    std::string depth_filename = "../videos/rgbd/scenes/desk/desk_1/desk_1_5_depth.png";    
    
    
    // Datos sacados del archivo desk_1.mat para el frame 5
    int im_c_left = 1;
    int im_c_right = 144;
    int im_r_top = 201;
    int im_r_bottom = 318;
    
    // Creo y levanto la imagen
    cv::Mat image;
    image = cv::imread(depth_filename, CV_LOAD_IMAGE_UNCHANGED);
    //cv::Size size = image.size(); size.height or size.width para tomar los valores

    // Verifico que haya leido la imagen correctamente
    if(!image.data){
        std::cout << "Hubo un error al cargar la imagen de profundidad" << std::endl;
        return -1;
    }
    
    // Busco los limites superiores e inferiores de x e y en la nube de puntos
    std::pair<float,float> cloudRC;
    unsigned short int depth;
    
    float r_top_limit    =  10.0; // Los inicializo al revés para que funcione bien al comparar
    float r_bottom_limit = -10.0;
    float c_left_limit   =  10.0;
    float c_right_limit  = -10.0;
    
    // TODO: paralelizar estos "for" usando funciones de OpenCV o PCL o lo que sea
    // Hint: que "from_flat_to_cloud" reciba una matriz (la imagen) directamente
    for(int r=im_r_top; r<=im_r_bottom; r++){
        for(int c=im_c_left; c<=im_c_right; c++){
            depth = image.at<unsigned short int>(r,c);
            
            cloudRC = from_flat_to_cloud(r, c, depth);
            
            if(cloudRC.first != -10000 and cloudRC.first < r_top_limit) r_top_limit = cloudRC.first;
            
            if(cloudRC.first != -10000 and cloudRC.first > r_bottom_limit) r_bottom_limit = cloudRC.first;
            
            if(cloudRC.second != -10000 and cloudRC.second < c_left_limit) c_left_limit = cloudRC.second;
            
            if(cloudRC.second != -10000 and cloudRC.second > c_right_limit) c_right_limit = cloudRC.second;
            
        }
    }
    
    
    /**
     * Levanto las nubes de puntos
     **/
     
    std::string cloud_in_filename  = "../videos/rgbd/scenes/desk/desk_1/desk_1_5.pcd";
    std::string cloud_out_filename = "../videos/rgbd/scenes/desk/desk_1/desk_1_7.pcd";
    
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
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    
    Eigen::Matrix4f transformation_matrix = icp.getFinalTransformation();
    std::cout << transformation_matrix << std::endl;
    
    show_transformation(filtered_cloud_in, transformation_matrix);

    return 0;
}
