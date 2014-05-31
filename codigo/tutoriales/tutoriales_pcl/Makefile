
ICP_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
ICP_FLAGS = -lpcl_io -lboost_system -lpcl_kdtree -lpcl_search -lpcl_common -lpcl_filters 

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lvtkCommon -lvtkFiltering -lpcl_visualization -lvtkRendering

OPENCV_INCLUDES = -I/usr/include/opencv
OPENCV_FLAGS = -lopencv_core -lopencv_highgui

NO_WARNINGS = -Wno-deprecated

all: one_iteration test


one_iteration:
	c++ $(NO_WARNINGS) one_iteration.cpp -o one_iteration $(ICP_INCLUDES) $(VISUALIZATION_INCLUDES) $(OPENCV_INCLUDES) $(ICP_FLAGS) $(VISUALIZATION_FLAGS) $(OPENCV_FLAGS)
	
test:
	c++ $(NO_WARNINGS) test.cpp -o test $(ICP_INCLUDES) $(OPENCV_INCLUDES) $(ICP_FLAGS) $(OPENCV_FLAGS)

clean:
	rm -f *.o one_iteration test
