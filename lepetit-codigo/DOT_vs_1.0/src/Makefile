export IPPROOT=/opt/intel/ipp/6.1.6.063/ia32
export INCLUDE=$(IPPROOT)/include:
export LD_LIBRARY_PATH=$(IPPROOT)/sharedlib:
export LIB=$(IPPROOT)/lib:
export CPATH=$(IPPROOT)/include:
export LIBRARY_PATH=$(IPPROOT)/lib:
export NLSPATH=$(IPPROOT)/lib/locale/%l_%t/%N:



OBJECTS = cv_utilities.o cv_camera.o cv_esm.o
DOT_INCLUDE = -I../include/
ESM_INCLUDE = -I../../resources/ESM/inc/
ESM_LIB = ../../resources/ESM/lib/ESMlib.a
OPENCV = `pkg-config --cflags --libs opencv`

.PHONY: clean

all: main2d main3d

main2d: main_dot2D.cc $(OBJECTS)
	g++ -o $@ -D MAIN_DOT_2D $^ $(ESM_LIB) $(DOT_INCLUDE) $(ESM_INCLUDE) $(OPENCV) -I$(IPPROOT)/include -L$(IPPROOT)/lib -L$(IPPROOT)/sharedlib -msse2 -lippi -lippcv -lippcore -lipps -lippm -lm


main3d: main_dot3D.cc $(OBJECTS)
	g++ -o $@ -D MAIN_DOT_3D $^ $(ESM_LIB) $(DOT_INCLUDE) $(ESM_INCLUDE) $(OPENCV) -I$(IPPROOT)/include -L$(IPPROOT)/lib -L$(IPPROOT)/sharedlib -msse2 -lippi -lippcv -lippcore -lipps -lippm -lm


cv_esm.o: cv_esm.cc
	g++ -c -o $@ $< $(ESM_LIB) $(DOT_INCLUDE) $(ESM_INCLUDE) $(OPENCV) -msse2

%.o: %.cc
	g++ -c -g -o $@ $< $(DOT_INCLUDE) $(OPENCV) -msse2

clean:
	rm -f *.o main2d main3d

