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
    pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*final);



    ICPResult res;//(icp.hasConverged(), icp.getFitnessScore(), final);
    res.has_converged = icp.hasConverged();
    res.score = icp.getFitnessScore();
    res.cloud = final;

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
