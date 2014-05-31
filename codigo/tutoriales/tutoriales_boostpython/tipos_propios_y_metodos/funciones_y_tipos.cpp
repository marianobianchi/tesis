#include <iostream>
#include "funciones_y_tipos.h"


FloatPair dame_par(std::string s, IntPair ip){

    std::cout << "El string que pasaste es: " << s << std::endl;
    
    return FloatPair((float) (ip.first * 2), ip.second / 2.0);
}


/**
 * Exporto todo a python
 **/

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

void export_all(){
    using namespace boost::python;
    
    class_<IntPair>("IntPair")
        .def(init<int,int>())
        .def_readwrite("first", &IntPair::first)
        .def_readwrite("second", &IntPair::second);
    
    class_<FloatPair>("FloatPair")
        .def(init<float,float>())
        .def_readwrite("first", &FloatPair::first)
        .def_readwrite("second", &FloatPair::second);        
    
    def("dame_par", dame_par);
}
