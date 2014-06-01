#include <iostream>

#include "seguimiento_common/tipos_basicos.h"
#include "icp_following.h"


int main(int argc, char** argv){
    
    // Datos sacados del archivo desk_1.mat para el frame 5
    int im_c_left = 1;
    int im_c_right = 144;
    int im_r_top = 201;
    int im_r_bottom = 318;
    IntPair topleft(im_r_top, im_c_left);
    IntPair bottomright(im_r_bottom, im_c_right);
    
    std::string depth_fname = "videos/rgbd/scenes/desk/desk_1/desk_1_5_depth.png";
    std::string source_cloud_fname  = "videos/rgbd/scenes/desk/desk_1/desk_1_5.pcd";
    std::string target_cloud_fname = "videos/rgbd/scenes/desk/desk_1/desk_1_7.pcd";
    
    DoubleIntPair topleft_bottomright = follow(topleft, bottomright, depth_fname, source_cloud_fname, target_cloud_fname);
    
    std::cout << "Top = "    << topleft_bottomright.first.first << std::endl;
    std::cout << "Bottom = " << topleft_bottomright.first.second << std::endl;
    std::cout << "Left = "   << topleft_bottomright.second.first << std::endl;
    std::cout << "Right = "  << topleft_bottomright.second.second << std::endl;

    return 0;

}
