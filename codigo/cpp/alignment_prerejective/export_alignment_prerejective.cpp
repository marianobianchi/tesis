#include "alignment_prerejective.h"

/*
 * Includes para exportar a python
 * */
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>



/*
 * Exporto todo a python
 * */

BOOST_PYTHON_MODULE(alignment_prerejective)
{

    using namespace boost::python;

    /*
     * Comparto para python lo minimo indispensable para usar
     * PointCloud's de manera razonable
     * */

    class_<APDefaults>("APDefaults")
        .def(init<>())
        .def_readwrite("leaf", &APDefaults::leaf)
        .def_readwrite("max_ransac_iters", &APDefaults::max_ransac_iters)
        .def_readwrite("points_to_sample", &APDefaults::points_to_sample)
        .def_readwrite("nearest_features_used", &APDefaults::nearest_features_used)
        .def_readwrite("simil_threshold", &APDefaults::simil_threshold)
        .def_readwrite("inlier_threshold", &APDefaults::inlier_threshold)
        .def_readwrite("inlier_fraction", &APDefaults::inlier_fraction)
        .def_readwrite("show_values", &APDefaults::show_values);

    class_<APResult>("APResult")
        .def_readwrite("has_converged", &APResult::has_converged)
        .def_readwrite("cloud", &APResult::cloud);

    def("align", alignment_prerejective);

}
