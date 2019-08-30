function config = exposeLoss(config, data, p)                   
% exposeLoss EXPOSE of the expLanes experiment bandwithExtension
%    config = exposeLoss(config, data, p)                       
%       config : expLanes configuration state                   
%       data : observations as a struct array                   
%       p : exposition parameters                               
                                                                
% Copyright: Mathieu Lagrange                                   
% Date: 24-Jun-2019                                             

% data.rawData{end}.loss

plot([data.rawData{end}.loss; data.rawData{end}.lossValidation]')
legend({'train' 'test'})   
ylabel('loss')
xlabel('epochs')

