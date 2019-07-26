function [config, store, obs] = baex2train(config, setting, data)
% baex2train TRAIN step of the expLanes experiment bandwithExtension
%    [config, store, obs] = baex2train(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

% Set behavior for debug mode
if nargin==0, bandwithExtension('do', 2, 'mask', {1 2 2 3 1:2}); return; else store=[]; obs=[]; end

switch setting.method
    case 'dnn'
        cc = config;
        if ~isempty(config.sequentialData) % sequential run
            data.modelPath =  config.sequentialData.modelPath;
            cc.step.setting.epochs = cc.step.setting.epochs-config.sequentialData.epochs;
        else
            if setting.epochs>10 % retrieve from data store to handle restart
                % retrieve previous model
                ss = num2cell(setting.infoId);
                ss{5} = ss{5}-1;
                ss = expStepSetting(config.factors, {ss}, 2);
                d = load([config.dataPath config.stepName{config.step.id} '/' ss.setting.infoHash  '_data']);
                o = load([config.dataPath config.stepName{config.step.id} '/' ss.setting.infoHash  '_obs']);
                data.modelPath =  d.modelPath{end};
                cc.step.setting.epochs = cc.step.setting.epochs-ss.setting.epochs;
                config.sequentialData.obs = o;
            else % init sequential data for storing model across epochs
                config.sequentialData.obs = [];
            end
        end
        [ss, obs] = expSystem(cc, data);
        store = data;
        store.modelPath = ss.modelPath;
        config.sequentialData.modelPath = store.modelPath{end};
        config.sequentialData.obs = expMerge(config.sequentialData.obs, obs);
        config.sequentialData.epochs = setting.epochs;
        obs = config.sequentialData.obs;
end