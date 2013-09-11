OBJECTS = cv_utilities.o cv_camera.o cv_esm.o
DOT_INCLUDE = -I../include/
ESM_INCLUDE = -I../../resources/ESM/inc/
ESM_LIB = ../../resources/ESM/lib/ESMlib.a
OPENCV = `pkg-config --cflags --libs opencv`

VARIABLES = $(shell . /opt/intel/ipp/6.1.6.063/ia32/tools/env/ippvars32.sh)

.PHONY: clean


main2d: main_dot2D.cc $(OBJECTS)
	g++ -o $@ $^ $(ESM_LIB) $(DOT_INCLUDE) $(ESM_INCLUDE) $(OPENCV) -I$(IPPROOT)/include -L$(IPPROOT)/lib -L$(IPPROOT)/sharedlib -msse2 -lippi -lippcv -lippcore -lipps -lippm -lm


cv_esm.o: cv_esm.cc
	g++ -c -o $@ $< $(ESM_LIB) $(DOT_INCLUDE) $(ESM_INCLUDE) $(OPENCV) -msse2

%.o: %.cc
	g++ -c -g -o $@ $< $(DOT_INCLUDE) $(OPENCV) -msse2

clean:
	rm -f *.o main2d

