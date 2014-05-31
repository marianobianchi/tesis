
ICP_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
ICP_FLAGS = -lpcl_io -lboost_system -lpcl_kdtree -lpcl_search -lpcl_common

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lflann_cpp -lflann_cpp-gd -lvtkCommon -lvtkFiltering -lpcl_visualization -lvtkRendering
VISUALIZATION_WARNINGS = -Wno-deprecated

main:
	c++ $(VISUALIZATION_WARNINGS) iterative_closest_point.cpp -o iterative_closest_point $(ICP_INCLUDES) $(VISUALIZATION_INCLUDES) $(ICP_FLAGS) $(VISUALIZATION_FLAGS)
	
	 
