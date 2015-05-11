
PCL_INCLUDES = -I/usr/local/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS = -lpcl_io -lpcl_filters			#-lpcl_kdtree -lpcl_search -lpcl_common -lboost_system 

PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python


NO_WARNINGS = -Wno-deprecated


all: cpp_main.so

cpp_main.so: cpp_main.o
	c++ -shared $^ $(BOOST_LIB) -o $@ $(PCL_FLAGS)

cpp_main.o: cpp_main.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PYTHON_INCLUDE) $(PCL_INCLUDES)

clean:
	rm -f *.o *.so
