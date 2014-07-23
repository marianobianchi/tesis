
PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python


NO_WARNINGS = -Wno-deprecated


all: cpp_main.so

cpp_main.so: cpp_main.o
	c++ -shared $^ $(BOOST_LIB) -o $@

cpp_main.o: cpp_main.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PYTHON_INCLUDE)

clean:
	rm -f *.o *.so
