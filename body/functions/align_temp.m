% just a dummy...
function[data] = align_temp(data)

action_names = fieldnames(data);
%      cut out idle where it does not belong!

n=3;
% for iaction=1:length(action_names)
% iaction = 3;
% action_name = action_names{iaction};
action_name = 'jump';
for isequence=1:size(data,2)
    idata = data(isequence).(action_name); % [3*15 x num_frames]
    X = idata;
    mean_shape = mean(X,2);
    X = bsxfun(@minus,X,mean_shape);
    if ~isempty(idata)
        num_frames = length(idata);
        [coeff,score,latent] = pca(X); % X = [3*15 x num_frames ] => coeff [ num_frames x * ]
        diff  = abs(coeff(2:num_frames,:)-coeff(1:num_frames-1,:));
        % scale from 0 to 1
        %         for i=1:n
        %             diff1 = diff(:,i);
        %             maxmin = (max(diff1)-min(diff1));
        %             sum(maxmin<10^-4)
        %             diff1 = (diff1-min(diff1))./maxmin;
        %             diff(:,i) = diff1;
        %         end
        %         index = find(diff(:,1)<0.05);
        m=mean(diff(:,1))
        index = find(diff(:,1)<0.05*m);
        zi = zeros(1,num_frames);
        zi(index) = m;
        
        figure(1), clf
        subplot(2,1,1), plot(abs(coeff(:,1:n))), axis on
        subplot(2,1,2),
        bar(diff(:,1:3)), axis on
        hold on
        plot(1:num_frames,zi,'r.','markersize',10)
        title(sprintf('sequence: %i ',isequence))
        %             plot(coeff(1:n,:)), axis on
        %             subplot(2,1,1), plot(coeff(:,1:n)), axis on
        %             subplot(2,1,2), plot(coeff(:,1:n)'), axis on
        pause(1)
        %             for iframe=1:num_frames
        %                 fprintf('(%i,%i,%i) of (%i,%i,*)\n',iaction,isequence,iframe,length(action_names),size(data,2))
        %                 shape  = reshape(idata(:,iframe),3,[]);
        %                 middle = mean(shape(:,index_inner),2); % [3x1]
        %                 shape  = bsxfun(@minus,shape,middle);
        %             end
    end
end
disp('temporal alignment done')

end