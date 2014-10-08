
% Variables
heigth = 480;
width = 640;
cant_frames = 2;
obj_name = 'desk';
number = '2';

% Nombres de archivos
video_frame_str = '{obj}_{number}_{nframe}_depth.png';
frames_path_str = 'scenes/{obj}/{obj}_{number}/';
pcd_frame_str = '{obj}_{number}_{nframe}.pcd';

infilename = strcat(frames_path_str, video_frame_str);
infilename = regexprep(infilename, '{obj}', obj_name);
infilename = regexprep(infilename, '{number}', number);

outfilename = strcat(frames_path_str, pcd_frame_str);
outfilename = regexprep(outfilename, '{obj}', obj_name);
outfilename = regexprep(outfilename, '{number}', number);


% Transformaciones
for i=1:cant_frames
    tmp_infilename = regexprep(infilename, '{nframe}', num2str(i));
    tmp_outfilename = regexprep(outfilename, '{nframe}', num2str(i));
    
    depth = imread(tmp_infilename);
    [pcloud distance] = depthToCloud(depth);
    
    savePcd(tmp_outfilename, pcloud);
end