
# c++ -I/usr/include/python2.7 -c hello.cpp -o hello.o
# c++ -shared hello.o -lboost_python -o hello.so


COMPILER = c++
PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python


.PHONY: clean

all: funciones_y_tipos.so

funciones_y_tipos.so: main.o funciones_y_tipos.o
	$(COMPILER) -shared $^ $(BOOST_LIB) -o $@

main.o: main.cpp
	$(COMPILER) $(PYTHON_INCLUDE) -c $^ -o $@

funciones_y_tipos.o: funciones_y_tipos.cpp
	$(COMPILER) $(PYTHON_INCLUDE) -c $^ -o $@

clean:
	rm -f *.o *.so
