heigth = 480;
width = 640;
cant_frames = 1;%98;
obj_name = 'desk';
number = '1';
video_frame_str = '{obj}_{number}_{nframe}_depth.png';
frames_path_str = 'scenes/{obj}/{obj}_{number}/';
pcd_frame_str = '{obj}_{number}_{nframe}.pcd';

infilename = strcat(frames_path_str, video_frame_str);
infilename = regexprep(infilename, '{obj}', obj_name);
infilename = regexprep(infilename, '{number}', number);

outfilename = strcat(frames_path_str, pcd_frame_str);
outfilename = regexprep(outfilename, '{obj}', obj_name);
outfilename = regexprep(outfilename, '{number}', number);


for i=1:cant_frames
    tmp_infilename = regexprep(infilename, '{nframe}', num2str(i));
    tmp_outfilename = regexprep(outfilename, '{nframe}', num2str(i));
    depth = imread(tmp_infilename);
    [pcloud distance] = depthToCloud(depth);
    
    pcloud_reshaped = reshape(pcloud, heigth*width, 3);
    
    [rows cols] = find(isnan(pcloud_reshaped));
    
    pcloud_reshaped(unique(rows),:) = [];
    
    savePcd(tmp_outfilename, pcloud);
end