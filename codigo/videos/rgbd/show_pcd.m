
data = readPcd('objs/coffee_mug/coffee_mug_5/coffee_mug_5_1_1.pcd');

figure
plot3(data(:,1),data(:,2),data(:,3),'.')
grid on