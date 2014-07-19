#ifndef __TIPOS_BASICOS__
#define __TIPOS_BASICOS__


#include <pcl/point_types.h>


typedef std::pair<float,float> FloatPair;
typedef std::pair<int,int> IntPair;

typedef std::pair<FloatPair,FloatPair > DoubleFloatPair;
typedef std::pair<IntPair,IntPair > DoubleIntPair;

struct ICPResult {
    //ICPResult(bool hc, float sc, pcl::PointCloud<pcl::PointXYZ> cloud) :
    //    has_converged(hc), score(sc), cloud(cloud) {}
    bool has_converged; // was found by icp?
    float score;        // icp score
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud; //point cloud

};


#endif //__TIPOS_BASICOS__
