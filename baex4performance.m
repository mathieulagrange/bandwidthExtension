function [config, store, obs] = baex4performance(config, setting, data)
% baex4performance PERFORMANCE step of the expLanes experiment bandwidthExtension
%    [config, store, obs] = baex4performance(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

% Set behavior for debug mode
if nargin==0, bandwidthExtension('do', 4, 'mask', {2 2 3 0 0 0 2 0 0 1 0 0 0 0 1}, 'debug', 0, 'parallel', 6); return; else store=[]; obs=[]; end

% load list of spectrogram files from step 1
d = expLoad(config, [], 1, 'data');
% if (strcmp(setting.split, 'train'))
%     d.testPath = d.trainPath;
%     d.testFiles = d.trainFiles;
% end

if exist(d.normFile, 'file')
    specNorm = readNPY(d.normFile);
else
    specNorm = [];
end

audioFileNames = {};

expRandomSeed();
saveIt = zeros(1, length(d.testFiles));
saveIt(randi(length(d.testFiles), 1, 30)) = 1;

options.destDir = '/tmp/';
options.segmentationFactor = 1;


parfor k=1:length(d.testFiles)
    sRefMag = readNPY(d.testFiles{k});
    sRefPhase = readNPY(strrep(d.testFiles{k}, '_magnitude.npy', '_phase.npy'));
    
    %     if k==1
    % mel projection matrices
    mel27 = fft2melmx((size(sRefMag, 2)-1)*2, d.samplingFrequency, 27);
    mel27 = mel27(:, 1:end/2+1);
    mel40 = fft2melmx((size(sRefMag, 2)-1)*2, d.samplingFrequency, 40);
    mel40 = mel40(:, 1:end/2+1);
    %     end
    
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
        case {'dnn', 'autoDense', 'autoStride'}
            hfMag = readNPY(data.predictions{k});
            hfMagSqueeze=[];
            for l=1:size(hfMag, 1)
                hfMagSqueeze(end+1:end+size(hfMag, 2), :) = squeeze(hfMag(l, :, :));
            end
            sPredMag(:, ceil(end/2+1):end) = hfMagSqueeze;
            
        case 'replicate'
            sPredMag(:, ceil(end/2+1):end) = replicationBaseline(sRefMag', setting.correlation)';
        case 'null'
            sPredMag(:, ceil(end/2+1):end)=0;
        case 'oracle'
    end
    sPredMag(sPredMag<0) = 0;
    if (isfield(config, 'debug') && config.debug)
        dbstop in baex4performance at 79
        clf
        iddg = randi(size(sRefMag, 1)-10);
        hold on
        plot(mean(sPredMag(iddg:iddg+10, :)), 'r')
        plot(mean(sRefMag(iddg:iddg+10, :)), 'k')
        legend({'prediction', 'reference'})
    end
    
    lossSpec(k) = immse(sRefMag, sPredMag);
    if ~isempty(specNorm)
        lossSpecNorm(k) = immse(sRefMag./repmat(specNorm, size(sRefMag, 1), 1), sPredMag./repmat(specNorm, size(sPredMag, 1), 1));
    else
        lossSpecNorm(k) = 0;
    end
    lossCqt27(k) = immse(mel27*sRefMag'.^2, mel27*sPredMag'.^2);
    lossCqt40(k) = immse(mel40*sRefMag'.^2, mel40*sPredMag'.^2);
    
    sRef = sRefMag.*exp(1i*sRefPhase);
    soundRef = ispecgram(sRef.');
    
    sPredPhase = 0;
    switch setting.estimatePhase
        case 'low'
            sPredPhase = sRefPhase;
            sPredPhase(:, ceil(end/2):end) = sRefPhase(:, 1:ceil(end/2));
        case 'mirror'
            sPredPhase = sRefPhase;
            sPredPhase(:, ceil(end/2):end) = sRefPhase(:, ceil(end/2):-1:1);
        case 'mirrorNegative'
            sPredPhase = sRefPhase;
            sPredPhase(:, ceil(end/2):end) = -sRefPhase(:, ceil(end/2):-1:1);
        case 'oracle'
            sPredPhase = sRefPhase;
        case 'gl'
            expRandomSeed();
            sPredPhase = randn(size(sPredMag));
            sPredPhase(:, 1:ceil(end/2)) = sRefPhase(:, 1:ceil(end/2));
            sPredPhase(:, ceil(end/2):end) = sRefPhase(:, 1:ceil(end/2));
            %         sPredPhase = sRefPhase;
            for l=1:setting.glNbIterations
                sPredPhase = angle(specgram(ispecgram(sPredMag.'.*exp(1i*sPredPhase.')), (size(sPredMag, 2)-1)*2)).';
                sPredPhase(:, 1:ceil(end/2)) = sRefPhase(:, 1:ceil(end/2));
            end
    end
    sPred = sPredMag.*exp(1i*sPredPhase);
    soundPred = ispecgram(sPred.');
    
    nmse(k) = immse(soundRef, soundPred)/(norm(soundRef, 2).^2/numel(soundRef));
    srr(k) = snr(soundRef, soundRef-soundPred);
    
%     if 1
%         fileNameRef = [expSave(config) '_' num2str(k) '_audio_ref.wav'];
%         fileNamePred = [expSave(config) '_' num2str(k) '_audio_pred.wav'];
%         audiowrite(fileNameRef, resample(soundRef, 48000, d.samplingFrequency), 48000);
%         audiowrite(fileNamePred, resample(soundPred, 48000, d.samplingFrequency), 48000);
%         res = PEASS_ObjectiveMeasure({fileNameRef}, fileNamePred, options);
%         ops(k) = res.OPS;
%         odg(k) =  PQevalAudio(fileNameRef, fileNamePred);
%         delete(fileNameRef);
%         delete(fileNamePred);
%     else
        ops(k) = NaN;
        odg(k) = NaN;
%     end

[mref,rfs]=pemo_internal(soundRef,d.samplingFrequency);
[mtest,rfs]=pemo_internal(soundPred,d.samplingFrequency);
psm(k) = pemo_metric(mref,mtest,rfs);
    
    if (setting.squeeze || saveIt(k))
        if length(soundPred)>d.samplingFrequency*60
            ls = floor(length(soundPred)/(d.samplingFrequency*60))*d.samplingFrequency*60;
            soundPreds = reshape(soundPred(1:ls), d.samplingFrequency*60, []);
        else
            soundPreds = soundPred;
        end
        fileName = [expSave(config) '_audio_'];
        idx=0;
        audioFileNames_k = {};
        for l=1:size(soundPreds, 2)
            fileNameL = [fileName num2str(k) '_'  num2str(idx) '.ogg'];
            audioFileNames_k{end+1} = fileNameL;
            sound = soundPreds(:, l)/max(abs(soundPreds(:, l)))*.9;
            audiowrite(fileNameL, sound, d.samplingFrequency);
            idx=idx+1;
        end
        audioFileNames{k} = audioFileNames_k;
    end
end

% obs.ops = ops;
% obs.odg = odg;
obs.psm = psm;
obs.lossCqt27 = lossCqt27;
obs.lossCqt40 = lossCqt40;
obs.lossSpec = lossSpec;
obs.lossSpecNorm = lossSpecNorm;
obs.nmse = nmse;
obs.srr = srr;

% obs.audioFileNames = {};
% for k=1:length(audioFileNames)
%     for l=1:length(audioFileNames{k})
%         tmp = audioFileNames{k};
%     obs.audioFileNames = {obs.audioFileNames{:} tmp{l} };
%     
%     end
% end

% disp('done');
