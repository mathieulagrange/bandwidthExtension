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
if nargin==0, bandwithExtension('do', 3, 'mask', {1 2 2 3 5 1}); return; else store=[]; obs=[]; end

% test on train
if (strcmp(setting.split, 'train'))
    data.testPath = data.trainPath;
    data.testFiles = data.trainFiles;
end

switch setting.method
    case 'dnn'
        if (setting.squeeze)
            % use model from non squeezed datast training
            ss = num2cell(setting.infoId);
            ss{2} = 1;
            ss = expStepSetting(config.factors, {ss}, 2);
            d = load([config.dataPath config.stepName{config.step.id-1} '/' ss.setting.infoHash  '_data']);
            data.modelPath = d.data.modelPath;
        end
        
        data.modelPath = data.modelPath{end};
        [ss, obs] = expSystem(config, data);
        store.predictions = ss.predictions;
end
