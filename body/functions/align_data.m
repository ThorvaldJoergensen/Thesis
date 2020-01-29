%
% input:
% -   W       : multiple body shapes in order: [3*N x 15]
% - Wtemplate : body shape [3x15]
function [W] = align_data(W, Wtemplate)

if ~isequal(size(Wtemplate),[3 15])
    error('%s: size of Wtemplate must be [3x15]',mfilename)
end

for f=1:3:size(W,1)
    [~,~,transform] = procrustes(Wtemplate(:,[1,8,10,13])',W(f:f+2,[1,8,10,13])',...
        'reflection',false);
    Z = transform.b*W(f:f+2,:)'*transform.T + repmat(transform.c(1,:),15,1);
    W(f:f+2,:)=Z';

%     if mod(f-1,1000)==0
%         disp(f)
%     end
end

end
