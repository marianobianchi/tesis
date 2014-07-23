#include <iostream>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include "tipos_basicos.h"
#include "rgbd.h"



void read_pcd(std::string pcd_filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_filename, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file\n");
        throw;
    }

    /*
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from pcd with the following fields: "
              << std::endl;

    for (size_t i = 0; i < cloud->points.size (); ++i){
        std::cout << "    " << cloud->points[i].x
                  << " "    << cloud->points[i].y
                  << " "    << cloud->points[i].z << std::endl;
    }*/
}


/**
 * Exporto todo a python
 **/

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

void export_all_rgbd(){
    using namespace boost::python;

    class_<ICPResult>("ICPResult")
        .def_readwrite("has_converged", &ICPResult::has_converged)
        .def_readwrite("score", &ICPResult::score)
        .def_readwrite("cloud", &ICPResult::cloud);
}
