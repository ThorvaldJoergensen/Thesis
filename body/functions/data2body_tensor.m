% read body data from files
function[data] = data2body_tensor()
dir_name     = '.\data_complete\';
% dir_name     = '~/data/body/';
folder_names = dir(dir_name);

data = struct();
min_num_frames = 10^5; % initial value
for iaction=1:length(folder_names)
    action_name = folder_names(iaction).name;
    idir_name   = [dir_name,action_name];
    if ~strcmp(idir_name(end),'.')
        if exist(idir_name,'dir')
            fprintf('%i: action >%s< \n',i,action_name)
            data(1).(action_name)=[];
            
            files     = dir([idir_name,'/*.mat']);
            num_files = length(files);
            fprintf('%i: action >%s< has %i examples\n',iaction,action_name,num_files)
            for isequence=1:num_files
                % fprintf('%i / %i \n',ii,num_files)
                filename = [idir_name,'/',files(isequence).name];
%                 disp(filename)
                O = load(filename);
                W = O.W;
                if iscell(W)
                    if length(W)>1
                        warning('cell has more entries')
                    end
                    W = W{1};
                end
                % W has size [3*num_frames x num_pts]
                [n_frames_x_3,num_pts] = size(W);% three rows => one shape
                num_frames = n_frames_x_3/3;
                W3 = zeros(3*num_pts,num_frames);
                for j=1:num_frames
                    k = 3*(j-1)+1;
                    shape   = W(k:k+2,:);
                    W3(:,j) = shape(:); % one shape = 3*15 points
                end
                data(isequence).(action_name) = W3;
                if num_frames<min_num_frames
                    min_num_frames = num_frames;
                end
            end
        end
    end
end
fprintf('minimum number of frames %i\n',min_num_frames)

end