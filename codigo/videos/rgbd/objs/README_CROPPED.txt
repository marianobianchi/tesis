RGB-D Object Dataset

The dataset contains 300 objects (aka instance) in 51 categories. The name of each object follows the format <category>_<number>, e.g. apple_1 is the first object (instance) in the "apple" category. Each instance consists of multiple RGB-D frames that come from three video sequences.

This part of the dataset contains the cropped RGB-D frames that tightly include the object, exactly as used in the object recognition evaluation of the paper introducing the RGB-D Object Dataset. There is another part of the dataset available containing 3D point clouds, in PCD format readable with the ROS Point Cloud Library (PCL), as well as a part containing the full 640x480 images from sensor.

Windows users: The dataset was compressed into a tarball using Linux. Some Windows extractors have problems reading the files. One program that can be used to extract the data in Windows is 7zip ( http://www.7-zip.org/ ). Open the single file extracted from the tarball again with 7zip to unpack it into a directory.

The files are named as follows:
<category>_<number>_<video>_<frame>_crop.png - A three-channel uint8 RGB image where pixels take on values between 0-255.
<category>_<number>_<video>_<frame>_depthcrop.png - A single-channel uint16 depth image. Each pixel gives the depth in millimeters, with 0 denoting missing depth. The depth image can be read using MATLAB with the standard function (imread), and in OpenCV by loading it into an image of type IPL_DEPTH_16U.
<category>_<number>_<video>_<frame>_maskcrop.png - A binary object segmentation mask.
<category>_<number>_<video>_<frame>_loc.txt - An ASCII file containing the location of the top left corner of the cropped images. The format is x,y where x is the horizontal pixel coordinate and y the vertical, starting from 1,1 at the top-left corner of the image. This file is necessary for converting cropped depth images into 3D point clouds (see "Depth Image To Point Cloud" at http://www.cs.washington.edu/rgbd-dataset/software.html).

Please cite the following paper if you use this dataset:

A Large-Scale Hierarchical Multi-View RGB-D Object Dataset 
Kevin Lai, Liefeng Bo, Xiaofeng Ren, and Dieter Fox 
IEEE International Conference on Robotics and Automation (ICRA), May 2011.

