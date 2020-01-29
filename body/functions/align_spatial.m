function[data] = align_spatial(data_orig)

% check on this function:
% -   W       : multiple body shapes in order: [3*N x 15]
% - Wtemplate : body shape [3x15]
% [W] = align_data(W, Wtemplate)

data   = data_orig;
shape0 = [];
% index_inner = [1,8]; % points 1 and 8 are body inner part
index_inner = [1,10, 13]; % points 1 and 8 are body inner part

action_names = fieldnames(data);
for iaction=1:length(action_names)
    action_name = action_names{iaction};
    %     idata       = data.(action_name);
    for isequence=1:size(data,2)
        idata = data(isequence).(action_name);
        %         index = round(linspace(1,size(idata,2),min_num_frames));
        % index = 1:min_num_frames; % first few frames
        if ~isempty(idata)
            num_frames = length(idata);
            for iframe=1:num_frames
                fprintf('(%i,%i,%i) of (%i,%i,*)\n',iaction,isequence,iframe,length(action_names),size(data,2))
                shape  = reshape(idata(:,iframe),3,[]); % [3 x 45]
                middle = mean(shape(:,index_inner),2); % [3x1]
                shape  = bsxfun(@minus,shape,middle);
                
                if isempty(shape0)
                    shape0 = shape;
                else
                    shape = align_data(shape,shape0);
                    triangle_static = shape0(:,index_inner);
                    triangle_deform = shape( :,index_inner);
                    [~,~,transform] = procrustes(triangle_static',triangle_deform','reflection',false);
                    shape_transformed = transform.b*shape'*transform.T + repmat(transform.c(1,:),15,1); % 15 x 3
                    shape = shape_transformed'; % [3x15]
                end
                idata(:,iframe) = shape(:);
                %                 shape  = reshape(idata(:,iframe),3,[]);
                %                 figure(1), clf
                %                 plot_body_shape(shape0,'r');
                %                 hold on
                %                 plot_body_shape(shape)
                %                 axis on
                %                 grid on
                %                 title(sprintf('frame %i',iframe))
                %                 view(180,20)
                %                 drawnow
            end
            data(isequence).(action_name) = idata;
        end
    end
end
disp('spatial alignment done')
end
