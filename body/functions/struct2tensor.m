function[T,labels] = struct2tensor(data,min_num_frames)
%  currently temporal selection included
action_names = fieldnames(data);

num_pts = 15;
icount  = 1;
T = zeros(3*num_pts,1,min_num_frames); % init tensor
labels = [];
index_inner = [1,8]; % points 1 and 8 are body inner part
for iaction=1:length(action_names)
    action_name = action_names{iaction};
%     idata = data.(action_name);
    for isequence=1:size(data,2)
        idata = data(isequence).(action_name);
        index = round(linspace(1,size(idata,2),min_num_frames));
        % index = 1:min_num_frames; % first few frames
        if ~isempty(idata)
            shapes = idata(:,index);
            for is = 1:size(min_num_frames)
                shape  = reshape(shapes(:,is),3,[]);
                %                 middle = mean(shape(:,index_inner),2); % [3x1]
                %                 shape = bsxfun(@minus,shape,middle);
                shapes(:,is) = shape(:);
            end
            T(:,icount,:) = shapes; % write matrix in tensor
            icount = icount+1;
            labels = [labels;iaction];
        end
    end
end
disp('sorting into tensor done')
size(T)

end
