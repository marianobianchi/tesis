
PCL_INCLUDES = -I/usr/local/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS = -lpcl_io -lboost_system -lpcl_common -lpcl_kdtree -lpcl_search -lpcl_filters -lpcl_features -lpcl_sample_consensus #-lpcl_registration -lpcl_segmentation

PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lvtkCommon -lvtkFiltering -lpcl_visualization -lvtkRendering #-lflann_cpp -lflann_cpp-gd

NO_WARNINGS = -Wno-deprecated


icp: icp_featured_based.cpp
	c++ $(NO_WARNINGS) $^ -o $@ $(PCL_INCLUDES) $(PCL_FLAGS)

clean:
	rm icp
