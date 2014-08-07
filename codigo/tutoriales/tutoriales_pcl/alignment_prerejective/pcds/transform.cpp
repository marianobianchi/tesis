#include <iostream>
#include <stdlib.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>


void showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " source_cloud.pcd transformed_cloud.pcd" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
    std::cout << "-td:  The translation distance in meters" << std::endl;
    std::cout << "-ta x | -ta y | -ta z:  The translation axis" << std::endl;
    std::cout << "-rd:  The rotation angle in radians" << std::endl;
    std::cout << "-ra x | -ra y | -ra z :  Rotation axis" << std::endl;
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

    if (filenames.size () != 2) {
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

    Eigen::Matrix4f r = Eigen::Matrix4f::Identity();

    
    float theta = M_PI / 4; // The angle of rotation in radians
    if(pcl::console::find_argument(argc, argv, "-rd") != -1){
        theta = atof(argv[pcl::console::find_argument(argc, argv, "-rd") + 1]);
    }
    std::cout << "rd = " << theta << std::endl;
    
    
    std::string rotation_axis = "x";
    if(pcl::console::find_argument(argc, argv, "-ra") != -1){
        rotation_axis = argv[pcl::console::find_argument(argc, argv, "-ra") + 1];
        if(rotation_axis != "x" and rotation_axis != "y" and rotation_axis != "z"){
            showHelp (argv[0]);
            return 0;
        }
    }
    std::cout << "ra = " << rotation_axis << std::endl;
    
    if(rotation_axis == "z"){
        // common rotation (z axis)
        r (0,0) = cos (theta);
        r (0,1) = -sin(theta);
        r (1,0) = sin (theta);
        r (1,1) = cos (theta);
    }

    if(rotation_axis == "x"){
        // common rotation (x axis)
        r (1,1) = cos (theta);
        r (1,2) = -sin(theta);
        r (2,1) = sin (theta);
        r (2,2) = cos (theta);
    }

    if(rotation_axis == "y"){
        // common rotation (y axis)
        r (0,0) = cos (theta);
        r (0,2) = sin(theta);
        r (2,0) = -sin (theta);
        r (2,2) = cos (theta);
    }
    
    float traslation_distance = 1.0;
    if(pcl::console::find_argument(argc, argv, "-td") != -1){
        traslation_distance = atof(argv[pcl::console::find_argument(argc, argv, "-td") + 1]);
    }
    std::cout << "td = " << traslation_distance << std::endl;
    
    std::string traslation_axis = "x";
    if(pcl::console::find_argument(argc, argv, "-ta") != -1){
        traslation_axis = argv[pcl::console::find_argument(argc, argv, "-ta") + 1];
        if(traslation_axis != "x" and traslation_axis != "y" and traslation_axis != "z"){
            showHelp (argv[0]);
            return 0;
        }
    }
    std::cout << "ta = " << traslation_axis << std::endl;

    if(traslation_axis == "y"){
        r (1,3) = traslation_distance;
    }
    else if(traslation_axis == "z"){
        r (2,3) = traslation_distance;
    }
    else{
        r (0,3) = traslation_distance;
    }

    // Print the transformation
    std::cout << "Transformation:" << std::endl;
    std::cout << r << std::endl;

    // Executing the transformation
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    pcl::transformPointCloud (*source_cloud, *transformed_cloud, r);

    pcl::io::savePCDFile (argv[filenames[1]], *transformed_cloud);
   
}
