#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

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
