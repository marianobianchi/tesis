PCL_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS = -lpcl_io -lpcl_common

PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python -lboost_system

NO_WARNINGS = -Wno-deprecated

share_matrix.so: export_matrix.o share_matrix.o
	c++ -shared $^ -o $@ $(BOOST_LIB) $(PCL_FLAGS)

share_matrix.o: share_matrix.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PCL_INCLUDES) $(PYTHON_INCLUDE)
	
export_matrix.o: export_matrix.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PCL_INCLUDES) $(PYTHON_INCLUDE)

all: share_matrix.so

clean:
	rm -f *.o *.so
