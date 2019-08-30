function [config, store, obs] = baex1features(config, setting, data)
% baex1features FEATURES step of the expLanes experiment bandwithExtension
%    [config, store, obs] = baex1features(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

% Set behavior for debug mode
if nargin==0, bandwithExtension('do', 1, 'mask', {0 2}); return; else store=[]; obs=[]; end

switch (setting.dataset)
    case 'librispeech'
        data.samplingFrequency = 5000;
        data.frameSize = 128;
    case 'gtzan'
        data.samplingFrequency = 22050;
        data.frameSize = 1024;
end

% launch computation of features python side
[store, obs] = expSystem(config, data);
store.samplingFrequency = data.samplingFrequency;