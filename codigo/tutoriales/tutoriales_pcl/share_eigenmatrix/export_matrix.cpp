#include "share_matrix.h"

/*
 * Includes para exportar a python
 * */
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>



/*
 * Exporto todo a python
 * */

BOOST_PYTHON_MODULE(share_matrix)
{

    using namespace boost::python;

    /*
     * Comparto para python lo minimo indispensable para usar
     * PointCloud's de manera razonable
     * */
    
    class_<FloatVector>("FloatVector")
        .def(vector_indexing_suite<FloatVector>() );
    
    class_<VectorMat>("VectorMat")
        .def(vector_indexing_suite<VectorMat>() );

    def("get_transformation", get_transformation);
    def("transform", transform);

}
