# rgbd.cpp makefile

OPENCV_INCLUDES = -I/usr/include/opencv
OPENCV_FLAGS = -lopencv_core -lopencv_highgui

ICP_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
ICP_FLAGS = -lpcl_io -lpcl_common

NO_WARNINGS = -Wno-deprecated


rgbd.o: rgbd.cpp
	c++ $(NO_WARNINGS) -c rgbd.cpp -o rgbd.o $(OPENCV_INCLUDES) $(ICP_INCLUDES)

clean:
	rm -f rgbd.o
