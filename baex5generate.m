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
if nargin==0, bandwithExtension('do', 5, 'mask', {1 2 1 0 0 0 1}); return; else store=[]; obs=[]; end

d = expLoad(config, [], 1, 'data');
reference=[];
idx = 0;
for k=1:length(d.testFiles)
    rk = readNPY(d.testFiles{k});
    reference((k-1)*500+1:(k-1)*500+size(rk, 1), :, :) = rk;
    rk_phase = readNPY(strrep(d.testFiles{k}, '.npy', '_phase.npy'));
    reference_phase((k-1)*500+1:(k-1)*500+size(rk, 1), :, :) = rk_phase;
    
    ref_spec = rk.*exp(1i*rk_phase);
    ref_spec2=[];
    for l=1:size(ref_spec)
        ref_spec2(end+1:end+size(ref_spec, 3), :) = squeeze(ref_spec(l, :, :))';
    end
    ref_sound = ispecgram(ref_spec2', setting.frameSize, setting.samplingFrequency);
    ls = floor(length(ref_sound)/(setting.samplingFrequency*60))*setting.samplingFrequency*60;
    sounds = reshape(ref_sound(1:ls), [], setting.samplingFrequency*60);
    
    for l=1:size(sounds, 1)
        fileName = [expSave(config) '_audio_' num2str(idx) '.wav'];
        sound = sounds(l, :)/max(abs(sounds(l, :)))*.9;
        audiowrite(fileName, sound, setting.samplingFrequency);
        idx=idx+1;
    end
end
