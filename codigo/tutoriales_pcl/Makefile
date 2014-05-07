
ICP_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
ICP_FLAGS = -lpcl_io -lboost_system -lpcl_kdtree -lpcl_search -lpcl_common -lpcl_filters 

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lvtkCommon -lvtkFiltering -lpcl_visualization -lvtkRendering

NO_WARNINGS = -Wno-deprecated

main:
	c++ $(NO_WARNINGS) one_iteration.cpp -o one_iteration $(ICP_INCLUDES) $(VISUALIZATION_INCLUDES) $(ICP_FLAGS) $(VISUALIZATION_FLAGS)
	
