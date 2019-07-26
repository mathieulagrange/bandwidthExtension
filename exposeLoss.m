function config = exposeLoss(config, data, p)                   
% exposeLoss EXPOSE of the expLanes experiment bandwithExtension
%    config = exposeLoss(config, data, p)                       
%       config : expLanes configuration state                   
%       data : observations as a struct array                   
%       p : exposition parameters                               
                                                                
% Copyright: Mathieu Lagrange                                   
% Date: 24-Jun-2019                                             


plot([data.rawData{1}.loss; data.rawData{1}.lossValidation]')
legend({'train' 'test'})   
ylabel('loss')
xlabel('epochs')
