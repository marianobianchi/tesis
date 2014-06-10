#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <boost/python.hpp>

#include "seguimiento_common/tipos_basicos.h"
#include "seguimiento_common/rgbd.h"
#include "icp_following.h"

BOOST_PYTHON_MODULE(icp_follow)
{
    using namespace boost::python;
    
    export_all_rgbd();
    export_follow();
}

