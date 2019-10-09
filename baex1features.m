function [config, store, obs] = baex1features(config, setting, data)
% baex1features FEATURES step of the expLanes experiment bandwidthExtension
%    [config, store, obs] = baex1features(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

% Set behavior for debug mode
if nargin==0, bandwidthExtension('do', 1, 'mask', {2 2}); return; else store=[]; obs=[]; end

switch (setting.dataset)
    case 'librispeech'
        data.samplingFrequency = 5000;
        data.frameSize = 128;
    case 'gtzan'
        data.samplingFrequency = 8000;
        data.frameSize = 256;
    case 'medleysolos'
        data.samplingFrequency = 8000;
        data.frameSize = 256;
end

% launch computation of features python side
[store, obs] = expSystem(config, data);
store.samplingFrequency = data.samplingFrequency;

% computation
sMin = inf;
sMax = 0;
for k=1:length(store.trainFiles)
    sRefMag = readNPY(store.trainFiles{k});
    s(k, :) = mean(mean(sRefMag, 3), 1);
    sMin = min(sMin, min(sRefMag(:)));
    sMax = max(sMax, max(sRefMag(:)));
end
sm = mean(s);
obs.dynamicOri = sMax-sMin;
% validation
nsMin = inf;
nsMax = 0;
for k=1:length(store.trainFiles)
    sRefMag = readNPY(store.trainFiles{k});
    nsRefMag = sRefMag./repmat(sm, size(sRefMag, 1), 1, size(sRefMag, 3));
    ns(k, :) = mean(mean(nsRefMag, 3), 1);
    nsMin = min(nsMin, min(nsRefMag(:)));
    nsMax = max(nsMax, max(nsRefMag(:)));
end
smn = mean(ns);
obs.dynamicNorm = nsMax-nsMin;

store.normFile = strrep(store.trainPath, 'train', 'norm.npy');
writeNPY(sm, store.normFile);
