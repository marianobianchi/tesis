#include "common.h"

/*
 * Includes para exportar a python
 * */
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>



/*
 * Exporto todo a python
 * */

BOOST_PYTHON_MODULE(common)
{

    using namespace boost::python;


    def("voxel_grid_downsample", voxel_grid_downsample);

}
