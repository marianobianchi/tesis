#coding=utf-8

from icp_follow import *

# Datos sacados del archivo desk_1.mat para el frame 5
im_c_left = 1;
im_c_right = 144;
im_r_top = 201;
im_r_bottom = 318;
topleft = IntPair(im_r_top, im_c_left);
bottomright = IntPair(im_r_bottom, im_c_right);
depth_fname = "videos/rgbd/scenes/desk/desk_1/desk_1_5_depth.png";
source_cloud_fname  = "videos/rgbd/scenes/desk/desk_1/desk_1_5.pcd";
target_cloud_fname = "videos/rgbd/scenes/desk/desk_1/desk_1_7.pcd";

# DoubleIntPair

topbottom_leftright = follow(topleft, bottomright, depth_fname, source_cloud_fname, target_cloud_fname);

print "Top should be 186, but is",      topbottom_leftright.first.first
print "Bottom should be 295, but is",   topbottom_leftright.first.second
print "Left should be 95, but is",      topbottom_leftright.second.first
print "Right should be 221, but is",    topbottom_leftright.second.second

