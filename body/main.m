
close all
clear all
clc
addpath(genpath('.'))

mat_filename = 'data_tensor.mat';
if exist(mat_filename,'file')
    fprintf('mat file exists => load data')
    O = load(mat_filename)
    T0 = O.T0; % original shapes
    T  = O.T; % shape-mean
    mean_shape = mean(mean(T0,2),3);

    labels         = O.labels;
    action_names   = O.action_names;
    min_num_frames = size(T,3);
else
    fprintf('no mat file => read and save data in mat file')
    %%% goal: create tensor of size [ 3 n_pts x num_seq x num_frames ]
    
    %%% (1) read in data
    [data] = data2body_tensor();
    disp('data reading done')
    
    %%% (2) spatial align
    [data] = align_spatial(data);
    % EXAMPLE.M ?
    
    %%% (3) temporal align/selection
    % test fun: needs to be redone
    % [data] = align_temp(data); % <- include into tensor function?
    
    %%% (4) sort data into tensor (INCLUDES temporal alignment currently)
    min_num_frames = 128;
    [T,labels] = struct2tensor(data,min_num_frames);
    
    % T = T(:,:,1:size(T,3)/2);
    % min_num_frames = size(T,3)/2;
    
    T0 = T;
    mean_shape = mean(mean(T0,2),3);
    T = T0-repmat(mean_shape,1,size(T0,2),size(T0,3));
    % T = T0(:,:,1:end-1)-T0(:,:,2:end); % difference tensor does not lead to good results
    % min_num_frames = min_num_frames-1;
    action_names = fieldnames(data);

    save(mat_filename,'T','T0','action_names','labels','mean_shape')
end

% %% part of tensor
% T = T(:,1:36,:);
% labels = labels(1:36);
% action_names=action_names(1:36);

[S,U_all,sv] = hosvd(T);
U1 = U_all{1};
U2 = U_all{2};
U3 = U_all{3};
% [S_full,U_all] = hosvd(T,[], [], true); % sign consistent version
% save('hosvd_results.mat','S','U1','U2','U3','sv','T','mean_shape')
disp('hosvd done')

%% plot some shapes
rows = 2;
cols = 6;
##for iframe = 1:min_num_frames
##    figure(1), clf
##    for iact=1:rows*cols % which action
##        shape=reshape(T0(:,iact,iframe),3,[]);
##        isub = iact;
##        subplot(rows,cols,isub)
##        plot_body_shape(shape)
##        %             title(action_names{labels(iact)})
##        axis equal
##    end
##    suptitle(sprintf('frame: %i',iframe))
##    drawnow
##    pause(0.1)
##end

%% plot cluster
close all
clc

ms = 30; % markersize
for idim=1:length(U_all)
    Ui = U_all{idim};
    %     Ui=Ui(:,4:6);
    figure(idim), clf
    if idim==2
        cmap = colormap(lines(length(unique(labels))));
        %         cmap = colormap(jet(length(unique(labels))));
        %         cmap = colormap(hsv(length(unique(labels))));
        cmap = cmap(labels,:);
        for icluster=1:length(action_names)
            hold on
            if icluster<8
                markertype = 'filled';
            else
                markertype = 'd';
            end
            scatter3(Ui(icluster==labels,1),...
                Ui(icluster==labels,2),...
                Ui(icluster==labels,3),ms,cmap(icluster==labels,:),...
                markertype,'DisplayName',action_names{icluster});
        end
        % plot all
        %         scatter3(Ui(:,1),Ui(:,2),Ui(:,3),ms,cmap,'filled','DisplayName','test');
        legend('show');
    else
        cmap = colormap(jet(size(Ui,1)));
        scatter3(Ui(:,1),Ui(:,2),Ui(:,3),ms,cmap,'filled');
    end
    grid on
end
if idim==3
    hold on
    plot3(Ui(:,1),Ui(:,2),Ui(:,3),'k-')
    for j=1:size(Ui,1)
        text(Ui(j,1),Ui(j,2),Ui(j,3),num2str(j))
    end
end
##
##%%
##%%%%%
##tensor_prod = @(S,U,n) tprod1(S, U, n);
##L = [45 25 128];
##U1 = U_all{1}(:,1:L(1));
##U2 = U_all{2}(:,1:L(2));
##U3 = U_all{3}(:,1:L(3));
##S = S(1:L(1),1:L(2),1:L(3));
##
##idim = 20;
##% iaction = 106;
##% iaction = 200;
##
##
##u2 = U2(idim,:);
##% iactions = [1,12];
##% u2 = mean(U2(iactions,:),1);
##figure(2), hold on
##plot3(u2(1),u2(2),u2(3),'rx','markersize',20)
##%%
##Th = squeeze(tensor_prod(tensor_prod(tensor_prod(S,U1,1),u2,2),U3,3));
##size(Th)
##num_frames = size(Th,2);
##
##xd = 20;
##zd = 45;
##figure(4),clf
##for idim=1:size(Th,2)
##    clf
##    shape_orig = reshape(T0(:,idim,idim),3,[]);
##    shape      = reshape(Th(:,idim)+mean_shape,3,[]);
##    plot_body_shape(shape_orig,'r');
##    hold on
##    plot_body_shape(shape);
##    suptitle(sprintf('action %s, frame %i ',action_names{labels(idim)},idim))
##    grid on
##    axis([-xd xd -xd xd -zd zd]);
##    drawnow
##    %     frame = getframe(gcf);
##    %     writeVideo(vid,frame);
##    pause(0.01)
##end
disp('done')

