
data = readPcd('scenes/desk/desk_1/desk_1_1.pcd');

figure
plot3(data(1:2:end,1),data(1:2:end,2),data(1:2:end,3),'*')
grid on