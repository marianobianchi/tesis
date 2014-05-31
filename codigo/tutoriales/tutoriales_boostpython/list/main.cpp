#include <iostream>

#include "list.h"

using namespace std;


/*
int main(int argc, char** argv){

    Lista l;
    l.agregarAdelante(3);
    l.agregarAdelante(2);
    l.agregarAdelante(1);
    
    cout << "El 1ro es el numero " << l.iesimo(0) << endl;
    cout << "El 2do es el numero " << l.iesimo(1) << endl;
    cout << "El 3ro es el numero " << l.iesimo(2) << endl;
    
    cout << l << endl;

}
*/


#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

using namespace boost::python;

BOOST_PYTHON_MODULE(list)
{
    class_<Lista>("Lista")
        .def(init<>())
        .def("add", &Lista::agregarAdelante)
        .def("sub", &Lista::iesimo)
    ;
}

