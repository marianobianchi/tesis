#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/pcl_visualizer.h>



void show_transformation(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud, Eigen::Matrix4f transformation_matrix)
{

	// Rotation matrix and translation vector
	// transformation_matrix

   	//			|-------> This column is the translation
	//	| a b c x |  \
	//	| d f g y |   }-> 3x3 matrix (rotation)
	//	| h i j z |  /
	//	| 0 0 0 1 |    -> We do not use this line

	// Defining a translation of 2 meters on the x axis.
	// transformation_matrix (0,3) = 2.5;

	// Printing the transformation
	printf ("\nThis is the rotation matrix :\n");
	printf ("    | %6.1f %6.1f %6.1f | \n", transformation_matrix (0,0), transformation_matrix (0,1), transformation_matrix (0,2));
	printf ("R = | %6.1f %6.1f %6.1f | \n", transformation_matrix (1,0), transformation_matrix (1,1), transformation_matrix (1,2));
	printf ("    | %6.1f %6.1f %6.1f | \n", transformation_matrix (2,0), transformation_matrix (2,1), transformation_matrix (2,2));
	printf ("\nThis is the translation vector :\n");
	printf ("t = < %6.1f, %6.1f, %6.1f >\n", transformation_matrix (0,3), transformation_matrix (1,3), transformation_matrix (2,3));

	// Executing the transformation
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());	// A pointer on a new cloud
	pcl::transformPointCloud (*source_cloud, *transformed_cloud, transformation_matrix);

	// Visualization
	printf("\nPoint cloud colors :	black	= original point cloud\n			red	= transformed point cloud\n");
	pcl::visualization::PCLVisualizer viewer ("Matrix transformation example");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (source_cloud, 0, 0, 0); // Where 255,255,255 are R,G,B colors
	viewer.addPointCloud (source_cloud, source_cloud_color_handler, "original_cloud");	// We add the point cloud to the viewer and pass the color handler

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 230, 20, 20);
	viewer.addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

	viewer.addCoordinateSystem (1.0, 0);
	viewer.setBackgroundColor(0.95, 0.95, 0.95, 0); // Setting background to a dark grey
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
	//viewer.setPosition(800, 400); // Setting visualiser window position

	while (!viewer.wasStopped ()) { // Display the visualiser untill 'q' key is pressed
		viewer.spinOnce ();
	}

}
