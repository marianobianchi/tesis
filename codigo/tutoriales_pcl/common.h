#include <pcl/filters/passthrough.h>


std::pair<float,float> from_flat_to_cloud(int imR, int imC, unsigned short int depth){

    // return something invalid if depth is zero
    if(depth == 0) return std::pair<float, float>(-10000, -10000);

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
    
    return std::pair<float, float>(cloud_row,cloud_col);

}


std::pair<int,int> from_cloud_to_flat(float cloud_row, float cloud_col, unsigned short int depth){

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
    
    return std::pair<int, int>(imR,imC);

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
