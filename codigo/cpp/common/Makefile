PCL_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS = -lpcl_common -lpcl_filters -lpcl_io -lpcl_registration -lpcl_segmentation -lpcl_kdtree -lpcl_search -lpcl_features

PYTHON_INCLUDE = -I/usr/include/python2.7
PYTHON_BOOST_FLAGS = -lboost_python

BOOST_FLAGS = -lboost_system

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lpcl_visualization

NO_WARNINGS = -Wno-deprecated


common.so: export_common.o common.o
	c++ -shared $^ -o $@ $(BOOST_FLAGS) $(PYTHON_BOOST_FLAGS) $(PCL_FLAGS) $(VISUALIZATION_FLAGS)
	cp $@ ..

common.o: common.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PCL_INCLUDES) $(VISUALIZATION_INCLUDES)
	
export_common.o: export_common.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PCL_INCLUDES) $(PYTHON_INCLUDE) $(VISUALIZATION_INCLUDES)


all: common.so

clean:
	rm -f *.o *.so
	rm -f ../common.so
