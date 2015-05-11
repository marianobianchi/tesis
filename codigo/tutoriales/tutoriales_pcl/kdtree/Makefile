
PCL_INCLUDES = -I/usr/local/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS = -lpcl_kdtree -lpcl_io -lpcl_visualization -lpcl_common

PYTHON_INCLUDE = -I/usr/include/python2.7

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lvtkCommon -lvtkFiltering -lvtkRendering

BOOST_FLAGS = -lboost_system

NO_WARNINGS = -Wno-deprecated


all: kdtree

kdtree: kdtree.cpp
	c++ $(NO_WARNINGS) $^ -o $@ $(PCL_INCLUDES) $(PYTHON_INCLUDE) $(VISUALIZATION_INCLUDES) $(PCL_FLAGS) $(VISUALIZATION_FLAGS) $(BOOST_FLAGS)

clean:
	rm kdtree.o kdtree
