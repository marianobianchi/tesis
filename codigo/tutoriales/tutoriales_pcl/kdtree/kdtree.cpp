#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <iostream>
#include <vector>
#include <ctime>

// Extra para el tutorial
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>


typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;


void showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " object.pcd" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
}


int main (int argc, char** argv)
{
    srand (time (NULL));

    PointCloud3D::Ptr cloud (new PointCloud3D);
    
    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
    
    if (filenames.size () != 1 || pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")){
        showHelp (argv[0]);
        return (1);
    }
  
    // Load object and scene
    pcl::console::print_highlight ("Loading point clouds...\n");
    if (pcl::io::loadPCDFile<Point3D> (argv[1], *cloud) < 0)
    {
        pcl::console::print_error ("Error loading object/scene file!\n");
        return (1);
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    kdtree.setInputCloud (cloud);

    Point3D searchPoint;

    searchPoint.x = 1.0 * rand () / (RAND_MAX + 1.0);
    searchPoint.y = 1.0 * rand () / (RAND_MAX + 1.0);
    searchPoint.z = 1.0 * rand () / (RAND_MAX + 1.0);

    // K nearest neighbor search

    int K = 10;

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    std::cout << "K nearest neighbor search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with K=" << K << std::endl;

    if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
    {
    for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
      std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
                << " " << cloud->points[ pointIdxNKNSearch[i] ].y 
                << " " << cloud->points[ pointIdxNKNSearch[i] ].z 
                << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
    }

    // Neighbors within radius search

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    float radius = 0.3;

    std::cout << "Neighbors within radius search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with radius=" << radius << std::endl;


    if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
    {
    for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
      std::cout << "    "  <<   cloud->points[ pointIdxRadiusSearch[i] ].x 
                << " " << cloud->points[ pointIdxRadiusSearch[i] ].y 
                << " " << cloud->points[ pointIdxRadiusSearch[i] ].z 
                << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
    }


    return 0;
}
