PCL_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS = -lpcl_io -lpcl_common

PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python -lboost_system

NO_WARNINGS = -Wno-deprecated

compute_centroid.so: compute_centroid.o
	c++ -shared $^ -o $@ $(BOOST_LIB) $(PCL_FLAGS)

compute_centroid.o: compute_centroid.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PCL_INCLUDES) $(PYTHON_INCLUDE)

clean:
	rm *.o *.so
