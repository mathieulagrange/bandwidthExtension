function [config, store, obs] = baex3test(config, setting, data)
% baex3test TEST step of the expLanes experiment bandwidthExtension
%    [config, store, obs] = baex3test(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

% Set behavior for debug mode
if nargin==0, bandwidthExtension('do', 3, 'mask', {2 2 1 2 3 3 5 1 1 0 0 0 1 1 1}); return; else store=[]; obs=[]; end

switch setting.method
    case 'dnn'
        if (setting.squeeze)
            % use model from non squeezed dataset training
            ss = num2cell(setting.infoId);
            ss{2} = 1;
            ss = expStepSetting(config.factors, {ss}, 2);
            unSqueezeDataFileName = [config.dataPath config.stepName{config.step.id-1} '/' ss.setting.infoHash  '_data.mat'];
            if (exist(unSqueezeDataFileName, 'file'))
            d = load(unSqueezeDataFileName);
            data.modelPath = d.data.modelPath;
            end
        end

        data.modelPath = data.modelPath{end};
        [ss, obs] = expSystem(config, data);
        store.predictions = ss.predictions;
end
