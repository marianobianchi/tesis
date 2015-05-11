
ICP_INCLUDES = -I/usr/local/include/pcl-1.7/ -I/usr/include/eigen3
ICP_FLAGS = -lpcl_registration -lpcl_search

PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lpcl_visualization

COMMON_FLAGS = -lpcl_filters

NO_WARNINGS = -Wno-deprecated


all: icp.so


icp.so: icp_following.o ../common/common.o
	c++ -shared $^ -o $@ $(BOOST_LIB) $(ICP_FLAGS) $(VISUALIZATION_FLAGS) $(COMMON_FLAGS)
	cp $@ ..

icp_following.o: icp_following.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(ICP_INCLUDES) $(PYTHON_INCLUDE) $(VISUALIZATION_INCLUDES)


clean:
	rm -f icp_following.o icp.so
	rm -f ../icp.so
