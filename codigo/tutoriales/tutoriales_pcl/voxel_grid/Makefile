NO_WARNINGS = -Wno-deprecated

PCL_INCLUDES = -I/usr/local/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS = -lpcl_io -lpcl_registration -lpcl_segmentation -lboost_system -lpcl_common -lpcl_kdtree -lpcl_filters -lpcl_search -lpcl_features

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lvtkCommon -lvtkFiltering -lpcl_visualization -lvtkRendering

voxel: voxel.cpp
	c++ $(NO_WARNINGS) $^ -o $@ $(PCL_INCLUDES) $(VISUALIZATION_INCLUDES) $(PCL_FLAGS) $(VISUALIZATION_FLAGS)
