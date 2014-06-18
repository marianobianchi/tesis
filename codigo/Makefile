
ICP_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
ICP_FLAGS = -lpcl_io -lboost_system -lpcl_kdtree -lpcl_search -lpcl_common -lpcl_filters 

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/
VISUALIZATION_FLAGS = -lvtkCommon -lvtkFiltering -lpcl_visualization -lvtkRendering

OPENCV_INCLUDES = -I/usr/include/opencv
OPENCV_FLAGS = -lopencv_core -lopencv_highgui

PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python


NO_WARNINGS = -Wno-deprecated


all: icp_follow.so


icp_follow.so: main.o icp_following.o rgbd.o
	c++ -shared $^ $(BOOST_LIB) -o $@ $(ICP_FLAGS) $(VISUALIZATION_FLAGS) $(OPENCV_FLAGS)

main.o: main.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(ICP_INCLUDES) $(VISUALIZATION_INCLUDES) $(OPENCV_INCLUDES) $(PYTHON_INCLUDE)
	
icp_following.o: icp_following.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(ICP_INCLUDES) $(PYTHON_INCLUDE)

rgbd.o: rgbd.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(OPENCV_INCLUDES) $(ICP_INCLUDES) $(PYTHON_INCLUDE)
	
clean:
	rm -f *.o main
