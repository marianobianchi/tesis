#ifndef __ICP_FOLLOWING__
#define __ICP_FOLLOWING__


/*
 * Includes para el codigo
 * */
#include <iostream>
#include <ctime>

#include <Eigen/Core>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>


#include "../common/common.h"

/*
 * Includes para exportar a python
 * */
#include <boost/python.hpp>


struct ICPResult {
    bool has_converged; // was found by icp?
    float score;        // icp score
    PointCloud3D::Ptr cloud;
    VectorMat transformation;
};

struct ICPDefaults {
    ICPDefaults(){
        euc_fit = -1.7976931348623157081e+308;
        max_corr_dist = 1.3407807929942595611e+154;
        max_iter = 10;
        transf_epsilon = 0;
        ran_iter = 0;
        ran_out_rej = 0.05;
        show_values = false;
    }

    double euc_fit;
    double max_corr_dist;
    int max_iter;
    double transf_epsilon;
    int ran_iter;
    double ran_out_rej;
    bool show_values;

};



ICPResult icp(PointCloud3D::Ptr source_cloud,
              PointCloud3D::Ptr target_cloud,
              ICPDefaults &icp_defaults);


#endif //__ICP_FOLLOWING__
