#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>


typedef pcl::PCLPointCloud2 VoxelPointCloud;
typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;


typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerNT;

int main (int argc, char** argv)
{
    PointCloud3D::Ptr cloud (new PointCloud3D);
    PointCloudNT::Ptr normalized_cloud (new PointCloudNT);
    PointCloudNT::Ptr cloud_filtered (new PointCloudNT);

    // Read cloud
    pcl::PCDReader reader;
    reader.read ("scene.pcd", *cloud);

    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height 
    << " data points (" << pcl::getFieldsList (*cloud) << ").\n";
    
    
    // Normalizing data
    normalized_cloud->points.resize(cloud->size());
    for (size_t i = 0; i < cloud->points.size(); i++) {
        normalized_cloud->points[i].x = cloud->points[i].x;
        normalized_cloud->points[i].y = cloud->points[i].y;
        normalized_cloud->points[i].z = cloud->points[i].z;
    }
    
    pcl::NormalEstimationOMP<PointNT,PointNT> nest;
    nest.setRadiusSearch (0.01);
    nest.setInputCloud (normalized_cloud);
    nest.compute (*normalized_cloud);

    //leaf
    float leaf = 0.01;
    if(pcl::console::find_argument(argc, argv, "-leaf") != -1){
        leaf = atof(argv[pcl::console::find_argument(argc, argv, "-leaf") + 1]);
    }
    std::cout << "leaf = " << leaf << std::endl;

    // Create the filtering object
    pcl::VoxelGrid<PointNT> sor;
    sor.setInputCloud (normalized_cloud);
    sor.setLeafSize (leaf, leaf, leaf);
    sor.filter (*cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
    << " data points (" << pcl::getFieldsList (*cloud_filtered) << ").\n";


    // Show
    pcl::visualization::PCLVisualizer visu("Downsampled");
    visu.addPointCloud (cloud_filtered, ColorHandlerNT (cloud_filtered, 0.0, 255.0, 0.0), "cloud_filtered");
    visu.spin ();


    return (0);
}
