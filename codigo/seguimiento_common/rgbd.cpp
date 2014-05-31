#include <iostream>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include "tipos_basicos.h"
#include "rgbd.h"


FloatPair from_flat_to_cloud(int imR, int imC, unsigned short int depth){

    // return something invalid if depth is zero
    if(depth == 0) return FloatPair(-10000, -10000);

    // cloud coordinates
    float cloud_row = (float) imR;
    float cloud_col = (float) imC;

	// images size is 640 COLS x 480 ROWS
	int rows_center = 240;
	int cols_center = 320;
    
    // focal distance
    float constant = 570.3;    
    
	// move the coordinate (0,0) from the top-left corner to the center 
    // of the plane
	cloud_row = cloud_row - rows_center;
    cloud_col = cloud_col - cols_center;
    
    
    // calculate cloud
    cloud_row = cloud_row * depth / constant / 1000;
    cloud_col = cloud_col * depth / constant / 1000;
    
    return FloatPair(cloud_row,cloud_col);

}


IntPair from_cloud_to_flat(float cloud_row, float cloud_col, float cloud_depth){

    unsigned short int depth = (unsigned short int) (cloud_depth * 1000 + 0.5);

    // images size is 640 COLS x 480 ROWS
	int rows_center = 240;
	int cols_center = 320;
    
    // focal distance
    float constant = 570.3;
    
    
    int imR, imC;
    
    // inverse of cloud calculation
    imR = (int) (cloud_row / depth * constant * 1000);
    imC = (int) (cloud_col / depth * constant * 1000);
    
    
    imR = imR + rows_center;
    imC = imC + cols_center;
    
    return IntPair(imR,imC);

}


DoubleFloatPair from_flat_to_cloud_limits(IntPair topleft, IntPair bottomright, std::string depth_filename)
{
    /**
     * Dada una imagen en profundidad y la ubicación de un objeto
     * en coordenadas sobre la imagen en profundidad, obtengo la nube
     * de puntos correspondiente a esas coordenadas y me quedo
     * unicamente con los valores máximos y mínimos de las coordenadas
     * "x" e "y" de dicha nube
     **/
    
    // Creo y levanto la imagen
    cv::Mat image;
    image = cv::imread(depth_filename, CV_LOAD_IMAGE_UNCHANGED);
    //cv::Size size = image.size(); size.height or size.width para tomar los valores

    // Verifico que haya leido la imagen correctamente
    assert(image.data); // std::cout << "Hubo un error al cargar la imagen de profundidad" << std::endl;
    
    // Busco los limites superiores e inferiores de x e y en la nube de puntos
    std::pair<float,float> cloudRC;
    unsigned short int depth;
    
    float r_top_limit    =  10.0; // Los inicializo al revés para que funcione bien al comparar
    float r_bottom_limit = -10.0;
    float c_left_limit   =  10.0;
    float c_right_limit  = -10.0;
    
    // TODO: paralelizar estos "for" usando funciones de OpenCV o PCL o lo que sea
    // Hint: que "from_flat_to_cloud" reciba una matriz (la imagen) directamente
    for(int r=topleft.first; r<=bottomright.first; r++){
        for(int c=topleft.second; c<=bottomright.second; c++){
            depth = image.at<unsigned short int>(r,c);
            
            cloudRC = from_flat_to_cloud(r, c, depth);
            
            if(cloudRC.first != -10000 and cloudRC.first < r_top_limit) r_top_limit = cloudRC.first;
            
            if(cloudRC.first != -10000 and cloudRC.first > r_bottom_limit) r_bottom_limit = cloudRC.first;
            
            if(cloudRC.second != -10000 and cloudRC.second < c_left_limit) c_left_limit = cloudRC.second;
            
            if(cloudRC.second != -10000 and cloudRC.second > c_right_limit) c_right_limit = cloudRC.second;
            
        }
    }
    
    return std::make_pair(std::make_pair(r_top_limit, r_bottom_limit),std::make_pair(c_left_limit, c_right_limit));
}

void read_pcd(std::string pcd_filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_filename, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file\n");
        throw;
    }
    
    /*
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from pcd with the following fields: "
              << std::endl;
    
    for (size_t i = 0; i < cloud->points.size (); ++i){
        std::cout << "    " << cloud->points[i].x
                  << " "    << cloud->points[i].y
                  << " "    << cloud->points[i].z << std::endl;
    }*/
}


void filter_cloud(  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud,
                    const std::string & field_name, 
                    const float & lower_limit, 
                    const float & upper_limit)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName (field_name);
    pass.setFilterLimits (lower_limit, upper_limit);
    
    // filter
    pass.filter(*filtered_cloud);
}






void export_all_rgbd(){

}
