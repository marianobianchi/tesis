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

    // Comparto el shared_ptr
    boost::python::register_ptr_to_python< boost::shared_ptr<PointCloud3D > >();

    // Comparto la clase point cloud
    class_<PointCloud3D >("PointCloudXYZ");

    class_<Point3D>("PointXYZ")
        .def_readonly("x", &Point3D::x)
        .def_readonly("y", &Point3D::y)
        .def_readonly("z", &Point3D::z);
        

    def("read_pcd", read_pcd);
    def("save_pcd", save_pcd);
    def("filter_cloud", filter_cloud);
    def("transform_cloud", transform_cloud);    
    
    def("points", points);
    def("get_point", get_point);
    def("show_clouds", show_clouds);
    def("voxel_grid_downsample", voxel_grid_downsample);
    def("filter_object_from_scene_cloud", filter_object_from_scene_cloud);

    class_<MinMax3D>("MinMax3D")
        .def_readonly("min_x", &MinMax3D::min_x)
        .def_readonly("min_y", &MinMax3D::min_y)
        .def_readonly("min_z", &MinMax3D::min_z)
        .def_readonly("max_x", &MinMax3D::max_x)
        .def_readonly("max_y", &MinMax3D::max_y)
        .def_readonly("max_z", &MinMax3D::max_z);

    def("get_min_max", get_min_max3D);

}
