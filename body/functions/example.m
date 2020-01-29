% example.m

seq = data(1).shoot;
[n_pts,nframes]=size(seq);
for i=1:nframes
    shape = reshape(seq(:,i),3,[]);
    m = mean(shape(:,[1,8]),2);
    
    figure(1), clf
    plot_body_shape(shape)
    hold on
    plot3(m(1),m(2),m(3),'r.','markersize',10)
    axis equal
    grid on
    title(sprintf('frame %i',i))
    drawnow
    pause(0.01)
end