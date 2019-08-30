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
if nargin==0, bandwithExtension('do', 2, 'mask', {1 2 1 2 1}, 'dryMode', 0); return; else store=[]; obs=[]; end

switch setting.method
    case 'dnn'
        cc = config;
        if ~isempty(config.sequentialData) % sequential run
            fprintf(2, 'sequential epoch\n');
            data.modelPath =  config.sequentialData.modelPath;
            cc.step.setting.epochs = cc.step.setting.epochs-config.sequentialData.epochs;
        else
            % default: init sequential data for storing model across epochs
            config.sequentialData.obs = [];
            if setting.epochs>10 % retrieve from data store to handle restart
                % retrieve previous model
                fprintf(2, 'restarting from previously computed epoch\n');
                ss = num2cell(setting.infoId);
                ss{5} = ss{5}-1;
                ss = expStepSetting(config.factors, {ss}, 2);
                fileName = [config.dataPath config.stepName{config.step.id} '/' ss.setting.infoHash];
                if (fopen([fileName  '_data.mat'])>0)
                    d = load([fileName  '_data']);
                    o = load([fileName  '_obs']);
                    data.modelPath =  d.modelPath{end};
                    cc.step.setting.epochs = cc.step.setting.epochs-ss.setting.epochs;
                    config.sequentialData.obs = o;
                end
            end
        end
        [ss, obs] = expSystem(cc, data);
        store = data;
        if (~isempty(ss))
            store.modelPath = ss.modelPath;
            if (~isempty(ss.modelPath))
                config.sequentialData.modelPath = store.modelPath{end};
                config.sequentialData.obs = expMerge(config.sequentialData.obs, obs);
                config.sequentialData.epochs = setting.epochs;
            end
            obs = config.sequentialData.obs;
        end
end