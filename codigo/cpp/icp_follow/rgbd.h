#ifndef __RGBD_METHODS__
#define __RGBD_METHODS__


void read_pcd(std::string pcd_filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

void export_all_rgbd();


#endif // __RGBD_METHODS__
