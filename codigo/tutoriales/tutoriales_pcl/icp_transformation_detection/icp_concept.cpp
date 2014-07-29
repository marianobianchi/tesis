#include <iostream>
#include <stdlib.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

// This function displays the help
void
showHelp(char * program_name)
{
  std::cout << std::endl;
  std::cout << "Usage: " << program_name << " cloud_filename.pcd [ -mcd max_corr_dist [ -mi max_iter [ -te trans_eps [ -ef euc_fit [ -re rot_eps]]]]]" << std::endl;
  std::cout << "-h:  Show this help." << std::endl;
}

// This is the main function
int
main (int argc, char** argv)
{

  // Show help
  if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
    showHelp (argv[0]);
    return 0;
  }

  // Fetch point cloud filename in arguments | Works with PCD and PLY files
  std::vector<int> filenames;
  bool file_is_pcd = false;

  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".ply");

  if (filenames.size () != 1)  {
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

    if (filenames.size () != 1) {
      showHelp (argv[0]);
      return -1;
    } else {
      file_is_pcd = true;
    }
  }

  // Load file | Works with PCD and PLY files
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

  if (file_is_pcd) {
    if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
      std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
      showHelp (argv[0]);
      return -1;
    }
  } else {
    if (pcl::io::loadPLYFile (argv[filenames[0]], *source_cloud) < 0)  {
      std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
      showHelp (argv[0]);
      return -1;
    }
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
  float theta = M_PI / 16; // The angle of rotation in radians
  
  // common rotation (z axis)
  //~ r (0,0) = cos (theta);
  //~ r (0,1) = -sin(theta);
  //~ r (1,0) = sin (theta);
  //~ r (1,1) = cos (theta);
  
  // common rotation (x axis)
  r (1,1) = cos (theta);
  r (1,2) = -sin(theta);
  r (2,1) = sin (theta);
  r (2,2) = cos (theta);
  
  // common rotation (y axis)
  //~ r (0,0) = cos (theta);
  //~ r (0,2) = sin(theta);
  //~ r (2,0) = -sin (theta);
  //~ r (2,2) = cos (theta);

  // Define a translation of 2.5 meters on the x axis.
  r (0,3) = 2.5;

  // Print the transformation
  printf ("Real transformation\n");
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
  //viewer.setPosition(800, 400); // Setting visualiser window position
  
  
  /*
   * Calculate ICP
   * */
   
   pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
   
   // Find ICP arguments
   if(pcl::console::find_argument(argc, argv, "-mcd") != -1){
      float max_corr_dist = atof(argv[pcl::console::find_argument(argc, argv, "-mcd") + 1]);
      std::cout << "mcd = " << max_corr_dist << std::endl;
      icp.setMaxCorrespondenceDistance (max_corr_dist);
   }
   
   if(pcl::console::find_argument(argc, argv, "-mi") != -1){
      int max_iter = atoi(argv[pcl::console::find_argument(argc, argv, "-mi") + 1]);
      std::cout << "mi = " << max_iter << std::endl;
      icp.setMaxCorrespondenceDistance (max_iter);
   }
   
   if(pcl::console::find_argument(argc, argv, "-te") != -1){
      float trans_eps = atof(argv[pcl::console::find_argument(argc, argv, "-te") + 1]);
      std::cout << "te = " << trans_eps << std::endl;
      icp.setMaxCorrespondenceDistance (trans_eps);
   }
   
   if(pcl::console::find_argument(argc, argv, "-ef") != -1){
      float euc_fit = atof(argv[pcl::console::find_argument(argc, argv, "-ef") + 1]);
      std::cout << "ef = " << euc_fit << std::endl;
      icp.setMaxCorrespondenceDistance (euc_fit);
   }
   
   //~ if(pcl::console::find_argument(argc, argv, "-re") != -1){
      //~ float rot_eps = atof(argv[pcl::console::find_argument(argc, argv, "-re") + 1]);
      //~ std::cout << "re = " << rot_eps << std::endl;
      //~ icp.setRotationEpsilon(rot_eps);
   //~ }
   
   
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
  
  
  // Agrego la nube al viewer
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> icp_cloud_color_handler (icp_found_cloud, 20, 20, 230); // Blue
  viewer.addPointCloud (icp_found_cloud, icp_cloud_color_handler, "icp_cloud");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "icp_cloud");
  
    
  
  while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
    viewer.spinOnce ();
  }

  return 0;
}
