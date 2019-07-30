function [config, store, obs] = baex5generate(config, setting, data)
% baex5generate GENERATE step of the expLanes experiment bandwithExtension
%    [config, store, obs] = baex5generate(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 24-Jun-2019

% Set behavior for debug mode
if nargin==0, bandwithExtension('do', 5, 'mask', {1 2 1 0 0 1 0 1}); return; else store=[]; obs=[]; end

d = expLoad(config, [], 1, 'data');

idx = 0;
for k=1:length(d.testFiles)
    sMag = readNPY(d.testFiles{k});
    sRefMag((k-1)*500+1:(k-1)*500+size(sMag, 1), :, :) = sMag;
    sPhase = readNPY(strrep(d.testFiles{k}, '.npy', '_phase.npy'));
    sRefPhase((k-1)*500+1:(k-1)*500+size(sPhase, 1), :, :) = sPhase;
end

sRef = sRefMag.*exp(1i*sRefPhase);
sPred = sRefMag;
sPred(:, ceil(end/2)+1:end, :) = data.hf;
sPred = sPred.*exp(1i*sRefPhase);

clear sRefPhase
clear sRefMag

sRefSqueeze=[];
sPredSqueeze=[];
for l=1:size(sRef, 1)
    sRefSqueeze(end+1:end+size(sRef, 3), :) = squeeze(sRef(l, :, :))';
    sPredSqueeze(end+1:end+size(sPred, 3), :) = squeeze(sPred(l, :, :))';
end
sRef = sRefSqueeze;
clear sRefSqueeze
sPred = sPredSqueeze;
clear sPredSqueeze


soundRef = ispecgram(sRef', setting.frameSize, setting.samplingFrequency);
clear sRef
soundPred = ispecgram(sPred', setting.frameSize, setting.samplingFrequency);
clear sPred

obs.mse = immse(soundRef, soundPred);
obs.srr = log10(sqrt(mean(soundRef.^2))/sqrt(obs.mse+eps));
clear soundRef

if (setting.squeeze)
    % save files only for squeezed dataset
    ls = floor(length(soundPred)/(setting.samplingFrequency*60))*setting.samplingFrequency*60;
    soundPreds = reshape(soundPred(1:ls), [], setting.samplingFrequency*60);
    
    fileName = [expSave(config) '_audio_'];
    for l=1:size(soundPreds, 1)
        sound = soundPreds(l, :)/max(abs(soundPreds(l, :)))*.9;
        audiowrite([fileName num2str(idx) '.wav'], sound, setting.samplingFrequency);
        idx=idx+1;
    end
end


