%%
% Levanto y muestro el PCD de la base
% FUNCIONA OK
data = readPcd('objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_1.pcd');
figure
plot3(data(:,1), data(:,2), data(:,3), '.')
grid on


%%
% Reproduzco los pasos segun la doc de la base para
% crear el mismo PCD de antes
% FUNCIONA OK
imagen = imread('objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_1_depthcrop.png');

mascara = uint16(imread('objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_1_maskcrop.png'));

[alto ancho] = size(imagen);


imagen = imagen.*mascara;


location = load('objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_1_loc.txt');

[pcloud distance] = depthToCloud(imagen, location);
pcloud = reshape(pcloud, alto*ancho, 3);

figure
plot3(pcloud(:,1), pcloud(:,3), pcloud(:,2)*-1, '.')
grid on

%%
% Hago lo mismo para la imagen de la escena del escritorio
% en el frame
%

im = imread('scenes/desk/desk_1/desk_1_5_depth.png');

top = 202;
bottom = 319;
left = 2;
right = 145;

cropim = im(top:bottom, left:right);

[alto ancho] = size(cropim);

loc = [left top];

[pcloud distance] = depthToCloud(cropim, loc);

repcloud = reshape(pcloud, alto*ancho, 3);

figure
plot3(repcloud(:,1), repcloud(:,3), repcloud(:,2)*-1, '.')
grid on

%%
source = readPcd('objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_1.pcd');
figure
plot3(source(:,1), source(:,2), source(:,3), '.')
grid on

target = readPcd('objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_4.pcd');
figure
plot3(target(:,1), target(:,2), target(:,3), '.')
grid on



%%
% Levanto y muestro el PCD recortado de una escena
% FUNCIONA OK
data2 = readPcd('../../pruebas_guardadas/desk_1_55_a_56_coffee_mug/object_cloud.pcd');



data_aligned = data2;
data_aligned(:,1) = data2(:,1) - mean(data2(not(isnan(data2(:,1))),1));
%data_aligned(:,2) = data2(:,2) - mean(data2(not(isnan(data2(:,2))),1));
%data_aligned(:,3) = data2(:,3) - mean(data2(not(isnan(data2(:,3))),1));


figure
plot3(data_aligned(:,1), data_aligned(:,2), data_aligned(:,3), '.')
grid on

%% Objetos que aparecen en una escena
load('scenes/table_small/table_small_1.mat');

[a nframes] = size(bboxes);

strings = {};
obj_count = 1;
for i=1:nframes
    [a nobjs] = size(bboxes{i});
    for o=1:nobjs
        strings{obj_count} = sprintf('%s_%i', bboxes{i}(o).category, bboxes{i}(o).instance);
        obj_count = obj_count + 1;
    end
end

strings = unique(strings)



