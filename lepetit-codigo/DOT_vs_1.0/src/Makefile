


OBJECTS = cv_utilities.o cv_camera.o cv_esm.o

DOT_INCLUDE = -I../include/

ESM_INCLUDE = -I../../resources/ESM/inc/


.PHONY: all clean


main2d: $(OBJECTS) #main_dot2D.cc


cv_esm.o: cv_esm.cc
	g++ -c -o $@ $< $(DOT_INCLUDE) $(ESM_INCLUDE) `pkg-config --cflags --libs opencv` -msse2


%.o: %.cc
	g++ -c -g -o $@ $< $(DOT_INCLUDE) `pkg-config --cflags --libs opencv` -msse2


