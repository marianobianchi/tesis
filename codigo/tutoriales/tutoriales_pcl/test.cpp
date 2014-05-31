#include <iostream>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/registration/icp.h>

#include "common.h"


using namespace std;


void test_flat_and_cloud_conversion(){
    pair<float, float> cloudXY;
    pair<int, int> flatXY;
    
    unsigned short int depth = 1300;
    
    for(int i=0;i<480;i++){
        for(int j=0;j<640;j++){
            cloudXY = from_flat_to_cloud(i, j, depth);
            flatXY = from_cloud_to_flat(cloudXY.first, cloudXY.second, depth);
            
            assert((flatXY.first -1) <= i <= (flatXY.first +1));
            assert((flatXY.second -1) <= j <= (flatXY.second +1));
        }
    }
}

void test_good_depth_to_cloud_conversion(){
    string depth_filename = "../videos/rgbd/scenes/desk/desk_1/desk_1_5_depth.png";    
    cv::Mat image;
    
    // Levanto la imagen
    image = cv::imread(depth_filename, CV_LOAD_IMAGE_UNCHANGED);

    // Verifico que haya leido la imagen correctamente
    assert(image.data);

    // Verifico que sea la profundidad requerida
    assert(image.depth() == CV_16U);
    
    // Verifico que el valor sea el esperado
    assert(image.at<unsigned short int>(0,0) == 1206);
}


int main(int argc, char** argv){

    test_flat_and_cloud_conversion();
    
    test_good_depth_to_cloud_conversion();
    

}



