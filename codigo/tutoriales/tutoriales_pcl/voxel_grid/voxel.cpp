#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>


typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandler3D;

int
main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ> ());

    // Read cloud
    pcl::PCDReader reader;
    reader.read ("scene.pcd", *cloud);

    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height 
    << " data points (" << pcl::getFieldsList (*cloud) << ").\n";

    //leaf
    float leaf = 0.01;
    if(pcl::console::find_argument(argc, argv, "-leaf") != -1){
        leaf = atof(argv[pcl::console::find_argument(argc, argv, "-leaf") + 1]);
    }
    std::cout << "leaf = " << leaf << std::endl;

    // Create the filtering object
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (leaf, leaf, leaf);
    sor.filter (*cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
    << " data points (" << pcl::getFieldsList (*cloud_filtered) << ").\n";


    // Show
    pcl::visualization::PCLVisualizer visu("Downsampled");
    visu.addPointCloud (cloud_filtered, ColorHandler3D (cloud_filtered, 0.0, 255.0, 0.0), "cloud_filtered");
    visu.spin ();


    return (0);
}
