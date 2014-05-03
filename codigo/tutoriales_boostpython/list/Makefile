
# Creado viendo la salida del comando "python setup.py build" usando el archivo 
# setup.py que hay en esta carpeta


# c++ -I/usr/include/python2.7 -c hello.cpp -o hello.o
# c++ -shared hello.o -lboost_python -o hello.so




COMPILER = c++
PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python


.PHONY: clean

all: list.so

# Este es el make para correr la pruebita que hay hecha en c++
#main: main.cpp
#	$(COMPILER) $< -o $@


list.so: main.o
	$(COMPILER) -shared $< $(BOOST_LIB) -o $@

main.o: main.cpp
	$(COMPILER) $(PYTHON_INCLUDE) -c $< -o $@


clean:
	rm -f *.o *.so
