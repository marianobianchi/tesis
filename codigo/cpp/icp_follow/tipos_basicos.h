#ifndef __TIPOS_BASICOS__
#define __TIPOS_BASICOS__


#include <pcl/point_types.h>


struct ICPResult {
    bool has_converged; // was found by icp?
    float score;        // icp score
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud; //point cloud

};


#endif //__TIPOS_BASICOS__
