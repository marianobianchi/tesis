
PCL_INCLUDES = -I/usr/include/pcl-1.7/ -I/usr/include/eigen3
PCL_FLAGS =      # -lpcl_io -lpcl_registration -lpcl_segmentation  -lpcl_common -lpcl_kdtree -lpcl_filters -lpcl_search -lpcl_features

PYTHON_INCLUDE = -I/usr/include/python2.7
BOOST_LIB = -lboost_python -lboost_system

VISUALIZATION_INCLUDES = -I/usr/include/vtk-5.8/

NO_WARNINGS = -Wno-deprecated


alignment_prerejective.so: export_alignment_prerejective.o alignment_prerejective.o
	c++ -shared $^ -o $@ $(BOOST_LIB) $(PCL_FLAGS)
	cp $@ ..

alignment_prerejective.o: alignment_prerejective.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PCL_INCLUDES) $(VISUALIZATION_INCLUDES)
	
export_alignment_prerejective.o: export_alignment_prerejective.cpp
	c++ $(NO_WARNINGS) -c $^ -o $@ $(PCL_INCLUDES) $(PYTHON_INCLUDE) $(VISUALIZATION_INCLUDES)

test: test.cpp alignment_prerejective.o
	c++ $(NO_WARNINGS) $^ -o $@ $(PCL_INCLUDES) $(PYTHON_INCLUDE) $(VISUALIZATION_INCLUDES) $(BOOST_LIB) $(PCL_FLAGS)

all: alignment_prerejective.so test

clean:
	rm -f export_alignment_prerejective.o alignment_prerejective.o alignment_prerejective.so
	rm ../alignment_prerejective.so
