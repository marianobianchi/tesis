
ICP_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
ICP_FLAGS = -lpcl_filters -lpcl_io

PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python


NO_WARNINGS = -Wno-deprecated


all: my_pcl.so


my_pcl.so: icp_following.o
	c++ -shared $^ -o $@ $(BOOST_LIB) $(ICP_FLAGS)
	cp $@ ..

icp_following.o: icp_following.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(ICP_INCLUDES) $(PYTHON_INCLUDE)


clean:
	rm -f icp_following.o my_pcl.so
	rm ../my_pcl.so
