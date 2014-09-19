#include <iostream>
#include <stdlib.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>


void showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " object.pcd scene.pcd" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
    std::cout << "-mcd:  Value passed to icp.setMaxCorrespondenceDistance" << std::endl;
    std::cout << "-mi:  Value passed to icp.setMaximumIterations" << std::endl;
    std::cout << "-te:  Value passed to icp.setTransformationEpsilon" << std::endl;
    std::cout << "-ef:  Value passed to icp.setEuclideanFitnessEpsilon \
                       (the maximum allowed distance error before the algorithm will be considered to have converged)" << std::endl;
}


int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr object (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene (new pcl::PointCloud<pcl::PointXYZ>);
    
    
    // Show help
    if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
        showHelp (argv[0]);
        return 0;
    }

    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    std::vector<int> filenames;

    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

    if (filenames.size () != 2) {
        showHelp (argv[0]);
        return -1;
    }

    // Load files
    if (pcl::io::loadPCDFile (argv[filenames[0]], *object) < 0 ||
            pcl::io::loadPCDFile (argv[filenames[1]], *scene) < 0)  {
        
        std::cout << "Error loading point clouds" << std::endl << std::endl;
        showHelp (argv[0]);
        return -1;
    }
    
    /*
    * Calculate ICP
    * */

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    // Find ICP arguments
    if(pcl::console::find_argument(argc, argv, "-mcd") != -1){
        float max_corr_dist = atof(argv[pcl::console::find_argument(argc, argv, "-mcd") + 1]);
        icp.setMaxCorrespondenceDistance (max_corr_dist);
    }
    std::cout << "mcd = " << icp.getMaxCorrespondenceDistance() << std::endl;

    if(pcl::console::find_argument(argc, argv, "-mi") != -1){
        int max_iter = atoi(argv[pcl::console::find_argument(argc, argv, "-mi") + 1]);
        
        icp.setMaximumIterations (max_iter);
    }
    std::cout << "mi = " << icp.getMaximumIterations() << std::endl;

    if(pcl::console::find_argument(argc, argv, "-te") != -1){
        float trans_eps = atof(argv[pcl::console::find_argument(argc, argv, "-te") + 1]);
        icp.setTransformationEpsilon (trans_eps);
    }
    std::cout << "te = " << icp.getTransformationEpsilon() << std::endl;

    if(pcl::console::find_argument(argc, argv, "-ef") != -1){
        float euc_fit = atof(argv[pcl::console::find_argument(argc, argv, "-ef") + 1]);
        icp.setEuclideanFitnessEpsilon (euc_fit);
    }
    std::cout << "ef = " << icp.getEuclideanFitnessEpsilon() << std::endl;

    icp.setInputSource(object);
    icp.setInputTarget(scene);
    pcl::PointCloud<pcl::PointXYZ>::Ptr icp_found_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*icp_found_cloud);

    Eigen::Matrix4f icp_transformation = icp.getFinalTransformation();

    // Print the transformation
    printf ("ICP transformation\n");
    std::cout << icp_transformation << std::endl;

    std::cout << "ICP score: " << icp.getFitnessScore() << std::endl;
    std::cout << "ICP converged: " << icp.hasConverged() << std::endl;


    // Visualizo el resultado
    printf("\nPoint cloud colors :  white  = original point cloud\n"
    "                        red  = transformed point cloud\n"
    "                       blue  = icp found point cloud\n");
    pcl::visualization::PCLVisualizer viewer ("ICP");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> object_color_handler (object, 255, 255, 255);
    viewer.addPointCloud (object, object_color_handler, "object");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_color_handler (scene, 230, 20, 20); // Red
    viewer.addPointCloud (scene, scene_color_handler, "scene");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> icp_cloud_color_handler (icp_found_cloud, 20, 20, 230); // Blue
    viewer.addPointCloud (icp_found_cloud, icp_cloud_color_handler, "icp_cloud");

    while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
        viewer.spinOnce ();
    }


 return (0);
}
