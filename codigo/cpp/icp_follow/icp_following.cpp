#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/registration/icp.h>

#include "tipos_basicos.h"
#include "rgbd.h"
#include "icp_following.h"

ICPResult follow (pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud)
{
    /**
     * Calculo ICP
     **/

    // Calculate icp transformation
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(object_cloud);
    icp.setInputTarget(target_cloud);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    /**
     * Busco los limites en el dominio de las filas y columnas del RGB
     * */
    int col_left_limit = 639;
    int col_right_limit = 0;
    int row_top_limit = 479;
    int row_bottom_limit = 0;

    IntPair flat_xy;

    for (int i = 0; i < Final.points.size (); i++){
        flat_xy = from_cloud_to_flat(Final.points[i].y, Final.points[i].x, Final.points[i].z);

        if(flat_xy.first < row_top_limit) row_top_limit = flat_xy.first;
        if(flat_xy.first > row_bottom_limit) row_bottom_limit = flat_xy.first;

        if(flat_xy.second < col_left_limit) col_left_limit = flat_xy.second;
        if(flat_xy.second > col_right_limit) col_right_limit = flat_xy.second;
    }

    ICPResult res;
    res.has_converged = icp.hasConverged();
    res.score = icp.getFitnessScore();
    int width = col_right_limit - col_left_limit;
    int height = row_bottom_limit - row_top_limit;
    res.size = width > height? width: height;
    res.top = row_top_limit;
    res.left = col_left_limit;

    return res;
}



/**
 * Exporto todo a python
 * */

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>


void export_follow(){

    using namespace boost::python;

    def("follow", follow);

}
