
ICP_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
ICP_FLAGS = -lpcl_io -lboost_system -lpcl_kdtree -lpcl_search -lpcl_common -lpcl_filters 

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lvtkCommon -lvtkFiltering -lpcl_visualization -lvtkRendering

OPENCV_INCLUDES = -I/usr/include/opencv
OPENCV_FLAGS = -lopencv_core -lopencv_highgui

NO_WARNINGS = -Wno-deprecated

all: main


main: main.cpp icp_following.o seguimiento_common/rgbd.o
	c++ $(NO_WARNINGS) $^ -o $@ $(ICP_INCLUDES) $(VISUALIZATION_INCLUDES) $(OPENCV_INCLUDES) $(ICP_FLAGS) $(VISUALIZATION_FLAGS) $(OPENCV_FLAGS)
	
icp_following.o: icp_following.cpp 
	c++ $(NO_WARNINGS) -c $^ -o $@ $(ICP_INCLUDES)

seguimiento_common/rgbd.o:
	$(MAKE) -C seguimiento_common
	
clean:
	rm -f *.o main
	$(MAKE) clean -C seguimiento_common
