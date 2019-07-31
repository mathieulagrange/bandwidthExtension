function [config, store, obs] = baex4performance(config, setting, data)
% baex4performance PERFORMANCE step of the expLanes experiment bandwithExtension
%    [config, store, obs] = baex4performance(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

% Set behavior for debug mode
if nargin==0, bandwithExtension('do', 4, 'mask', {1 1 2 3 5 1}); return; else store=[]; obs=[]; end

% load list of spectrogram files from step 1
d = expLoad(config, [], 1, 'data');
if (strcmp(setting.split, 'train'))
    d.testPath = d.trainPath;
    d.testFiles = d.trainFiles;
end

% mel projection matrices
mel27 = fft2melmx(setting.frameSize, setting.samplingFrequency, 27);
mel27 = mel27(:, 1:end/2+1);
mel40 = fft2melmx(setting.frameSize, setting.samplingFrequency, 40);
mel40 = mel40(:, 1:end/2+1);

idx = 0;
for k=1:length(d.testFiles)
    sRefMag = readNPY(d.testFiles{k});
    sRefPhase = readNPY(strrep(d.testFiles{k}, '.npy', '_phase.npy'));
    
    sRefMagSqueeze=[];
    sRefPhaseSqueeze=[];
    for l=1:size(sRefMag, 1)
        sRefMagSqueeze(end+1:end+size(sRefMag, 3), :) = squeeze(sRefMag(l, :, :))';
        sRefPhaseSqueeze(end+1:end+size(sRefPhase, 3), :) = squeeze(sRefPhase(l, :, :))';
    end
    sRefMag = sRefMagSqueeze;
    sRefPhase = sRefPhaseSqueeze;   
    %    sRefMag  = reshape(permute(sRefMag, [1 3 2]), size(sRefMag, 1)*size(sRefMag, 3), size(sRefMag, 2));
    %    sRefPhase  = reshape(permute(sRefPhase, [1 3 2]), size(sRefPhase, 1)*size(sRefPhase, 3), size(sRefPhase, 2));
    
    sPredMag = sRefMag;
    switch setting.method
        case 'dnn'
            hfMag = readNPY(data.predictions{k});   
            hfMagSqueeze=[];
            for l=1:size(hfMag, 1)
                hfMagSqueeze(end+1:end+size(hfMag, 2), :) = squeeze(hfMag(l, :, :));
            end
            sPredMag(:, ceil(end/2+1):end) = hfMagSqueeze;
            
        case 'replication'
            sPredMag(:, ceil(size(sRefMag, 2)/2)+1:size(sRefMag, 2)) = replicationBaseline(sRefMag', setting.correlation)';
        case 'null'
            sPredMag(:, ceil(size(sRefMag, 2)/2)+1:size(sRefMag, 2))=0;
        case 'oracle'
    end
    if (isfield(config, 'debug'))
        dbstop in baex4performance at 70
        clf
        iddg = randi(size(sRefMag, 1));
        plot([mean(sRefMag(iddg:iddg+10, :)); mean(sPredMag(iddg:iddg+10, :))]')
        legend({'reference', 'prediction'})
    end
    
    obs.lossSpec(k) = immse(sRefMag, sPredMag);
    obs.lossCqt27(k) = immse(mel27*sRefMag'.^2, mel27*sPredMag'.^2);
    obs.lossCqt40(k) = immse(mel40*sRefMag'.^2, mel40*sPredMag'.^2);
    
    sRef = sRefMag.*exp(1i*sRefPhase);
    sPred = sPredMag.*exp(1i*sRefPhase);
    soundRef = ispecgram(sRef.');
    soundPred = ispecgram(sPred.');
    
    obs.mse(k) = immse(soundRef, soundPred);
    obs.srr(k) = log10(sqrt(mean(soundRef.^2))/sqrt(obs.mse(k)+eps));
    
    if (setting.squeeze)
        % save files only for squeezed dataset
        ls = floor(length(soundPred)/(setting.samplingFrequency*60))*setting.samplingFrequency*60;
        soundPreds = reshape(soundPred(1:ls), setting.samplingFrequency*60, []);
        
        fileName = [expSave(config) '_audio_'];
        for l=1:size(soundPreds, 2)
            sound = soundPreds(:, l)/max(abs(soundPreds(:, l)))*.9;
            audiowrite([fileName num2str(idx) '.wav'], sound, setting.samplingFrequency);
            idx=idx+1;
        end
    end
end