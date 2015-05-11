NO_WARNINGS = -Wno-deprecated

PCL_INCLUDES = -I/usr/local/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS = -lpcl_io -lboost_system -lpcl_common #-lpcl_registration   -lpcl_kdtree -lpcl_search

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lpcl_visualization -lvtkCommon -lvtkFiltering -lpcl_visualization -lvtkRendering

pcdvideo: pcdvideo.cpp
	c++ $(NO_WARNINGS) $^ -o $@ $(PCL_INCLUDES) $(VISUALIZATION_INCLUDES) $(PCL_FLAGS) $(VISUALIZATION_FLAGS)
