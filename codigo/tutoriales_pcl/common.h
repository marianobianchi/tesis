

std::pair<float,float> from_flat_to_cloud(int imX, int imY, float depth){

    // cloud coordinates
    float cx = (float) imX;
    float cy = (float) imY;

	// images size is 640 COLS x 480 ROWS
	int rows_center = 240;
	int cols_center = 320;
    
    // focal distance
    float constant = 570.3;
    
    // convert depth from m to mm
    depth *= 1000;
    
	// move the coordinate (0,0) from the top-left corner to the center 
    // of the plane
	cx -= cols_center;
    cy -= rows_center;
    
    
    // calculate cloud
    cx = cx * depth / constant / 1000;
    cy = cy * depth / constant / 1000;
    
    return std::pair<float, float>(cx,cy);

}


std::pair<int,int> from_cloud_to_flat(float cx, float cy, float depth){

    // images size is 640 COLS x 480 ROWS
	int rows_center = 240;
	int cols_center = 320;
    
    // focal distance
    float constant = 570.3;
    
    
    int imX, imY;
    
    // inverse of cloud calculation
    imX = (int) (cx / depth * constant);
    imY = (int) (cy / depth * constant);
    
    
    imX += cols_center;
    imY += rows_center;
    
    return std::pair<int, int>(imX,imY);

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
