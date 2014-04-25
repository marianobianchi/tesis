RGB-D Scenes Dataset

This dataset contains 8 scenes annotated with objects that belong to the RGB-D Object Dataset. Each scene is a single video sequence consisting of multiple RGB-D frames.

Windows users: The dataset was compressed into a tarball using Linux. Some Windows extractors have problems reading the files. One program that can be used to extract the data in Windows is 7zip ( http://www.7-zip.org/ ). Open the single file extracted from the tarball again with 7zip to unpack it into a directory.

The files are named as follows:
<scene>_<frame>.png - A three-channel uint8 RGB image where pixels take on values between 0-255.
<scene>_<frame>_depth.png - A single-channel uint16 depth image. Each pixel gives the depth in millimeters, with 0 denoting missing depth. The depth image can be read using MATLAB with the standard function (imread), and in OpenCV by loading it into an image of type IPL_DEPTH_16U. Software for converting depth images to 3D point clouds is available at http://www.cs.washington.edu/rgbd-dataset/software.html.

There is also a single MATLAB .mat file per scene containing the ground truth annotations. bboxes is a cell array where each element is a video frame. The element is empty if there are no objects in the video frame. Otherwise, it will be a struct array where each element is one object annotation. For example, to get the second object annotation from frame 4, you access bboxes{4}(2). The annotation includes the category name, instance number, and the four corners of the bounding rectangle.


Please cite the following paper if you use this dataset:

A Large-Scale Hierarchical Multi-View RGB-D Object Dataset 
Kevin Lai, Liefeng Bo, Xiaofeng Ren, and Dieter Fox 
IEEE International Conference on Robotics and Automation (ICRA), May 2011.

