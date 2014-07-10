FLAGS = -lboost_system -lboost_python
INCLUDES = -I/usr/include/python2.7

NO_WARNINGS = -Wno-deprecated

all: depth_to_rgb.so


depth_to_rgb.so: depth2RGB.o
	c++ -shared $^ $(FLAGS) -o $@
	cp $@ ..

depth2RGB.o: depth2RGB.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(INCLUDES)

clean:
	rm -f *.o