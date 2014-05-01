% La idea era ver si la informacion de xIm y yIm en los
% pcd de los objetos de la base se correspondian a los
% x,y de la imagen plana de profundidad (los archivos de 16 bits)
% Comprobamos que no, que se corresponden a los de la imagen en RGB

xgrid1 = data(:,5);
ygrid1 = data(:,6);
depth = data(:,3)*1000;
%
center = [320 240];
topleft = [1 1];
% [imh, imw] = size(depth);
constant = 570.3; %dist focal
MM_PER_M = 1000;

% convert depth image to 3d point clouds
% pcloud = zeros(imh,imw,3);
% xgrid = ones(imh,1)*(1:imw) + (topleft(1)-1) - center(1);
% ygrid = (1:imh)'*ones(1,imw) + (topleft(2)-1) - center(2);


xgrid = xgrid1 + (topleft(1)-1) - center(1);
ygrid = ygrid1 + (topleft(2)-1) - center(2);


pcloud = zeros(size(xgrid,1),3);

pcloud(:,1) = xgrid.*depth/constant/MM_PER_M;
pcloud(:,2) = ygrid.*depth/constant/MM_PER_M;
pcloud(:,3) = depth/MM_PER_M;
distance = sqrt(sum(pcloud.^2,3));
