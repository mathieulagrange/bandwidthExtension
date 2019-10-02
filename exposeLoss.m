function config = exposeLoss(config, data, p)                   
% exposeLoss EXPOSE of the expLanes experiment bandwithExtension
%    config = exposeLoss(config, data, p)                       
%       config : expLanes configuration state                   
%       data : observations as a struct array                   
%       p : exposition parameters                               
                                                                
% Copyright: Mathieu Lagrange                                   
% Date: 24-Jun-2019                                             

% data.rawData{end}.loss
loss = [];
lossValidation = [];
for k=1:length(data.rawData)
   loss = [loss data.rawData{k}.loss(end-min(length(data.rawData{k}.loss)-1, 9):end)];
   lossValidation = [lossValidation data.rawData{k}.lossValidation(end-min(length(data.rawData{k}.lossValidation)-1, 9):end)];
end

% plot([data.rawData{end}.loss; data.rawData{end}.lossValidation]')
plot([lossValidation]')
% legend({'loss' 'lossValidation'})   
ylabel('loss')
xlabel('epochs')

config.displayData.plotSuccess = 1;

