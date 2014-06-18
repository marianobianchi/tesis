#ifndef __RGBD_METHODS__
#define __RGBD_METHODS__

FloatPair from_flat_to_cloud(int imR, int imC, unsigned short int depth);


IntPair from_cloud_to_flat(float cloud_row, float cloud_col, float cloud_depth);


DoubleFloatPair from_flat_to_cloud_limits(IntPair topleft, IntPair bottomright, std::string depth_filename);


void read_pcd(std::string pcd_filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

void filter_cloud(  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud,
                    const std::string & field_name, 
                    const float & lower_limit, 
                    const float & upper_limit);


void export_all_rgbd();


#endif // __RGBD_METHODS__
