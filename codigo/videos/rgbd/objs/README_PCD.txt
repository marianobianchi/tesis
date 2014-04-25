RGB-D Object Dataset

The dataset contains 300 objects (aka instance) in 51 categories. The name of each object follows the format <category>_<number>, e.g. apple_1 is the first object (instance) in the "apple" category. Each instance consists of multiple RGB-D frames that come from three video sequences.

This part of the dataset contains the 3D point clouds of views of each object, in PCD format readable with the ROS Point Cloud Library (PCL). There is a part of the dataset available containing cropped images of the objects for extracting visual features, and another part of the dataset containing the full 640x480 images from sensor.

Windows users: The dataset was compressed into a tarball using Linux. Some Windows extractors have problems reading the files. One program that can be used to extract the data in Windows is 7zip ( http://www.7-zip.org/ ). Open the single file extracted from the tarball again with 7zip to unpack it into a directory.

The files are named as follows:
<category>_<number>_<video>_<frame>.pcd - A 3D point cloud stored in PCD format, readable with the ROS Point Cloud Library (PCL). Each point is stored with 6 fields: the 3D coordinate (x, y, z), the color packed into 24 bits with 8 bits per channel (rgb), and the image coordinate of the point (imX, imY). Note: In the PCD files, for a given point (x, y, z), the x-axis points to the right along the image plane, y points into the image plane, and z points upwards along the image plane. Software for reading PCD files is available at http://www.cs.washington.edu/rgbd-dataset/software.html.

Please cite the following paper if you use this dataset:

A Large-Scale Hierarchical Multi-View RGB-D Object Dataset 
Kevin Lai, Liefeng Bo, Xiaofeng Ren, and Dieter Fox 
IEEE International Conference on Robotics and Automation (ICRA), May 2011.

