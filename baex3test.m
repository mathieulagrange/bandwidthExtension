function [config, store, obs] = baex3test(config, setting, data)
% baex3test TEST step of the expLanes experiment bandwithExtension
%    [config, store, obs] = baex3test(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

% Set behavior for debug mode
if nargin==0, bandwithExtension('do', 3, 'mask', {1 1 2 3 4 1 1}); return; else store=[]; obs=[]; end

if (strcmp(setting.split, 'train'))
    data.testPath = data.trainPath;
    data.testFiles = data.trainFiles;
end

% if (setting.squeezeTest ~= setting.squeeze)
%     dMask = config.mask;
%     dMask{2} = setting.squeezeTest;
%     config = toto;
%     d = expLoad(config, [], 'data');
%     data.testPath = d.trainPath;
%     data.testFiles = d.trainFiles;
% end

% [status,cmdout] =
switch setting.method
    case 'dnn'
        data.modelPath = data.modelPath{end};
        [ss, obs] = expSystem(config, data);
        store.predictions = ss.predictions;
end
