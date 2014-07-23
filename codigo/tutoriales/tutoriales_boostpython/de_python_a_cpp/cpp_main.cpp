#include <boost/python.hpp>


void mostrar(boost::python::object o){

    o.attr("show")();

}


BOOST_PYTHON_MODULE(cpp_main)
{
    using namespace boost::python;    
    
    def("mostrar", mostrar);
}
