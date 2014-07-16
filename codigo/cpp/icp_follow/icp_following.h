#ifndef __ICP_FOLLOWING__
#define __ICP_FOLLOWING__


ICPResult follow (IntPair top_left,
                  IntPair bottom_right,
                  std::string depth_fname,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud);

void export_follow();

#endif //__ICP_FOLLOWING__
