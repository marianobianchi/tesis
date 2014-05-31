
# Creado viendo la salida del comando "python setup.py build" usando el archivo 
# setup.py que hay en esta carpeta


# c++ -I/usr/include/python2.7 -c hello.cpp -o hello.o
# c++ -shared hello.o -lboost_python -o hello.so




COMPILER = c++
OBJECTS = hello.o
PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python


.PHONY: clean

all: hello.so


hello.so: hello.o
	$(COMPILER) -shared $< $(BOOST_LIB) -o $@

hello.o: hello.cpp
	$(COMPILER) $(PYTHON_INCLUDE) -c $< -o $@


clean:
	rm -f *.o *.so
