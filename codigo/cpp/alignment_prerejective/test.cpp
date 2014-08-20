#include <iostream>

#include "alignment_prerejective.h"



int main(int argc, char** argv){

    PointCloud3D::Ptr object(new PointCloud3D);
    pcl::io::loadPCDFile<Point3D> ("object.pcd", *object);
    
    PointCloud3D::Ptr scene(new PointCloud3D);
    pcl::io::loadPCDFile<Point3D> ("scene.pcd", *scene);
    
    APDefaults ap_defaults;
    ap_defaults.show_values = true;
    
    APResult ap_result = alignment_prerejective(object, scene, ap_defaults);
    
    if(ap_result.has_converged){
        std::cout << "CONVERGIO!" << std::endl;
    }
    else{
        std::cout << "NO CONVERGIO!" << std::endl;
    }

    return 0;
}
