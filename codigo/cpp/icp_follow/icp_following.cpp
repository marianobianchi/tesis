#include <iostream>

#include <pcl/registration/icp.h>

#include "icp_following.h"

ICPResult follow (boost::python::object source_cloud,
                  boost::python::object target_cloud)
{
    /**
     * Calculo ICP
     **/

    // Calculate icp transformation
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud);
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
