function show_pcd(data)
    figure
    plot3(data(1:end,1),data(1:end,2),data(1:end,3),'.')
    grid on
end