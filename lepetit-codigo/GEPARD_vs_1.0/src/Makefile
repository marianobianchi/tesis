export IPPROOT=/opt/intel/ipp/6.1.6.063/ia32
export INCLUDE=$(IPPROOT)/include:
export LD_LIBRARY_PATH=$(IPPROOT)/sharedlib:
export LIB=$(IPPROOT)/lib:
export CPATH=$(IPPROOT)/include:
export LIBRARY_PATH=$(IPPROOT)/lib:
export NLSPATH=$(IPPROOT)/lib/locale/%l_%t/%N:



OBJECTS = cv_camera.o  cv_esm.o  cv_gepard.o  cv_harris.o  cv_homography.o  cv_hyper.o  cv_pcabase.o  cv_utilities.o
PROJ_INCLUDE = -I../include/
ESM_INCLUDE = -I../../resources/ESM/inc/
ESM_LIB = ../../resources/ESM/lib/ESMlib.a
OPENCV = `pkg-config --cflags --libs opencv`

.PHONY: clean

all: main

main: main_wil.cc $(OBJECTS)
	g++ -o $@ $^ $(ESM_LIB) $(PROJ_INCLUDE) $(ESM_INCLUDE) $(OPENCV) -I$(IPPROOT)/include -L$(IPPROOT)/lib -L$(IPPROOT)/sharedlib -msse2 -lippi -lippcv -lippcore -lipps -lippm -lm


cv_esm.o: cv_esm.cc
	g++ -c -o $@ $(ESM_LIB) $< $(PROJ_INCLUDE) $(ESM_INCLUDE) $(OPENCV) -msse2

%.o: %.cc
	g++ -c -g -o $@ $< $(PROJ_INCLUDE) $(OPENCV) -msse2

clean:
	rm -f *.o main

