% plot a 3d body shape
%
% input:
% - shape   [ 3 x 15 ]
%
function plot_body_shape(shape,col)

if nargin<2
    col = [0  0.447  0.741];
end
if ~isequal(size(shape),[3,15])
    error('%s: unexpected input size for shape.',mfilename)
end
bool_text = false; % test case
% bool_text = true; % test case

persistent bone_map
if isempty(bone_map)
    bone_map =...
        [
        1     2
        2     3
        3     4
        1     5
        5     6
        6     7
        1     8
        8    10
        10    11
        11    12
        8    13
        13    14
        14    15
        8     9];
end
shape = shape([1 3 2],:);

% plot3(shape(1,:),shape(2,:),shape(3,:),'go','LineWidth',2,'MarkerSize',10);
% plot3(shape(1,:),shape(2,:),shape(3,:),'co','LineWidth',1,'MarkerSize',8,...
%     'MarkerEdgeColor','k','MarkerFaceColor','c');
plot3(shape(1,:),shape(2,:),shape(3,:),'co','LineWidth',1,'MarkerSize',8,...
    'MarkerEdgeColor','k','MarkerFaceColor',col);
hold on
plot3([shape(1,bone_map(:,1)); shape(1,bone_map(:,2))],...
    [shape(2,bone_map(:,1)); shape(2,bone_map(:,2))],...
    [shape(3,bone_map(:,1)); shape(3,bone_map(:,2))],...
    'LineWidth',2,'color',col)
axis equal

if bool_text
    for i=1:size(shape,2)
        text(shape(1,i),shape(2,i),shape(3,i),num2str(i))
    end
end
end