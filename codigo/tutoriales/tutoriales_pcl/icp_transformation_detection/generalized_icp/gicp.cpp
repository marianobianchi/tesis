#include <iostream>
#include <stdlib.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/registration/gicp.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>


void showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " cloud_filename.pcd" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
    std::cout << "-mcd:  Value passed to icp.setMaxCorrespondenceDistance" << std::endl;
    std::cout << "-mi:  Value passed to icp.setMaximumIterations" << std::endl;
    std::cout << "-te:  Value passed to icp.setTransformationEpsilon" << std::endl;
    std::cout << "-ef:  Value passed to icp.setEuclideanFitnessEpsilon " <<
                 "(the maximum allowed distance error before the algorithm will be considered to have converged)" << std::endl;
    std::cout << "-tr:  The translation in X axis in meters" << std::endl;
    std::cout << "-pi:  The rotation angle will be MATH_PI / THIS_VALUE" << std::endl;
    std::cout << "-r x | -r y | -r z :  Axis of rotation" << std::endl;
    std::cout << "-re:  Value passed to icp.setRotationEpsilon" << std::endl;
}

int main (int argc, char** argv)
{

    // Show help
    if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
        showHelp (argv[0]);
        return 0;
    }

    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    std::vector<int> filenames;


    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

    if (filenames.size () != 1) {
        showHelp (argv[0]);
        return -1;
    }

    // Load file
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
        std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
        showHelp (argv[0]);
        return -1;
    }


    /**
    * Rotation from axis and angle (no funciona)
    * 
    * r(0,0) = cos(theta) + std::pow(ux,2) * (1 - cos(theta));
    * r(0,1) = ux * yx * (1 - cos(theta)) - uz * sin(theta);
    * r(0,2) = ux * uz * (1 - cos(theta)) + uy * sin(theta);
    * r(1,0) = uy * ux * (1 - cos(theta)) + uz * sin(theta);
    * r(1,1) = cos(theta) * std::pow(uy,2) * (1 - cos(theta));
    * r(1,2) = uy * uz * (1 - cos(theta)) - ux * sin(theta);
    * r(2,0) = uz * ux * (1 - cos(theta)) + uy * sin(theta);
    * r(2,1) = uz * uy * (1 - cos(theta)) + ux * sin(theta);
    * r(2,2) = cos(theta) + std::pow(uz,2) * (1 - cos(theta));
    **/

    Eigen::Matrix4f r = Eigen::Matrix4f::Identity();

    // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
    int parts_of_pi = 4;
    if(pcl::console::find_argument(argc, argv, "-pi") != -1){
        int poi = atoi(argv[pcl::console::find_argument(argc, argv, "-pi") + 1]);
        std::cout << "pi = " << poi << std::endl;
        parts_of_pi = poi;
    }
    float theta = M_PI / parts_of_pi; // The angle of rotation in radians
    
    
    std::string axis = "x";
    if(pcl::console::find_argument(argc, argv, "-r") != -1){
        axis = argv[pcl::console::find_argument(argc, argv, "-r") + 1];
        if(axis != "x" and axis != "y" and axis != "z"){
            showHelp (argv[0]);
            return 0;
        }
        std::cout << "r = " << axis << std::endl;
    }
    if(axis == "z"){
        // common rotation (z axis)
        r (0,0) = cos (theta);
        r (0,1) = -sin(theta);
        r (1,0) = sin (theta);
        r (1,1) = cos (theta);
    }

    if(axis == "x"){
        // common rotation (x axis)
        r (1,1) = cos (theta);
        r (1,2) = -sin(theta);
        r (2,1) = sin (theta);
        r (2,2) = cos (theta);
    }

    if(axis == "y"){
        // common rotation (y axis)
        r (0,0) = cos (theta);
        r (0,2) = sin(theta);
        r (2,0) = -sin (theta);
        r (2,2) = cos (theta);
    }

    // Define a translation of 2.5 meters on the x axis.
    r (0,3) = 1;
    if(pcl::console::find_argument(argc, argv, "-tr") != -1){
        r (0,3) = atof(argv[pcl::console::find_argument(argc, argv, "-tr") + 1]);
    }

    // Print the transformation
    std::cout << "Real transformation: " << "Rotation axis (radians) = " << axis << "(PI / " << parts_of_pi << ")" << std::endl;
    std::cout << r << std::endl;

    // Executing the transformation
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    pcl::transformPointCloud (*source_cloud, *transformed_cloud, r);

    // Visualization
    printf(  "\nPoint cloud colors :  white  = original point cloud\n"
    "                        red  = transformed point cloud\n"
    "                       blue  = icp found point cloud\n");
    pcl::visualization::PCLVisualizer viewer ("Matrix transformation example");

    // Define R,G,B colors for the point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (source_cloud, 255, 255, 255);
    // We add the point cloud to the viewer and pass the color handler
    viewer.addPointCloud (source_cloud, source_cloud_color_handler, "original_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 230, 20, 20); // Red
    viewer.addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

    //viewer.addCoordinateSystem (1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");


    /*
    * Calculate ICP
    * */
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

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

    if(pcl::console::find_argument(argc, argv, "-re") != -1){
        float rot_eps = atof(argv[pcl::console::find_argument(argc, argv, "-re") + 1]);
        icp.setRotationEpsilon(rot_eps);
    }
    std::cout << "re = " << icp.getRotationEpsilon() << std::endl;


    icp.setInputSource(source_cloud);
    icp.setInputTarget(transformed_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr icp_found_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*icp_found_cloud);

    Eigen::Matrix4f icp_transformation = icp.getFinalTransformation();

    // Print the transformation
    printf ("ICP transformation\n");
    std::cout << icp_transformation << std::endl;

    std::cout << "ICP score: " << icp.getFitnessScore() << std::endl;
    std::cout << "ICP converged: " << icp.hasConverged() << std::endl;

    pcl::transformPointCloud (*source_cloud, *icp_found_cloud, icp_transformation);

    // Agrego la nube al viewer
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> icp_cloud_color_handler (icp_found_cloud, 20, 20, 230); // Blue
    viewer.addPointCloud (icp_found_cloud, icp_cloud_color_handler, "icp_cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "icp_cloud");



    while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
        viewer.spinOnce ();
    }

    return 0;
}
