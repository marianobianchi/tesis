# Es un makefile reciclado del makefile originado por CMAKE


CXX_FLAGS = -Wno-deprecated -I/usr/include/vtk-5.8 -I/usr/include/pcl-1.7 -I/usr/include/eigen3 -I/usr/include/ni -march=native -msse4.2 -mfpmath=sse -Wno-invalid-offsetof

CXX_DEFINES = -DEIGEN_USE_NEW_STDVECTOR -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET



all:
    g++ $(CXX_DEFINES) $(CXX_FLAGS) -o pcd_write.o -c pcd_write.cpp
    g++ -Wno-deprecated pcd_write.o -o pcd_write_test -rdynamic -lboost_system-mt -lboost_filesystem-mt -lboost_thread-mt -lpthread -lboost_date_time-mt -lboost_iostreams-mt -lboost_serialization-mt -lpcl_common -lOpenNI /usr/lib/libvtkCommon.so.5.8.0 /usr/lib/libvtkRendering.so.5.8.0 /usr/lib/libvtkHybrid.so.5.8.0 /usr/lib/libvtkCharts.so.5.8.0 -lpcl_io -lboost_system-mt -lboost_filesystem-mt -lboost_thread-mt -lpthread -lboost_date_time-mt -lboost_iostreams-mt -lboost_serialization-mt -lpcl_common -lpcl_octree -lboost_system-mt -lboost_filesystem-mt -lboost_thread-mt -lpthread -lboost_date_time-mt -lboost_iostreams-mt -lboost_serialization-mt -lpcl_common -lOpenNI -lpcl_io -lpcl_octree /usr/lib/libvtkViews.so.5.8.0 /usr/lib/libvtkInfovis.so.5.8.0 /usr/lib/libvtkWidgets.so.5.8.0 /usr/lib/libvtkHybrid.so.5.8.0 /usr/lib/libvtkParallel.so.5.8.0 /usr/lib/libvtkVolumeRendering.so.5.8.0 /usr/lib/libvtkRendering.so.5.8.0 /usr/lib/libvtkGraphics.so.5.8.0 /usr/lib/libvtkImaging.so.5.8.0 /usr/lib/libvtkIO.so.5.8.0 /usr/lib/libvtkFiltering.so.5.8.0 /usr/lib/libvtkCommon.so.5.8.0 -lm /usr/lib/libvtksys.so.5.8.0 -ldl -Wl,-rpath,/usr/lib/openmpi/lib -Wl,-rpath-link,/usr/lib/openmpi/lib
