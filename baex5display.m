function [config, store, obs] = baex5display(config, setting, data)
% baex5display DISPLAY step of the expLanes experiment bandwidthExtension
%    [config, store, obs] = baex5display(config, setting, data)
%      - config : expLanes configuration state
%      - setting   : set of factors to be evaluated
%      - data   : processing data stored during the previous step
%      -- store  : processing data to be saved for the other steps
%      -- obs    : observations to be saved for analysis

% Copyright: Mathieu Lagrange
% Date: 16-Oct-2019

% Set behavior for debug mode
if nargin==0, bandwidthExtension('do', 5, 'mask', {3 2 [1 2 5 6] 3 3 2 3 2 1 1 1 1 1 1 1}, 'debug', 1); return; else store=[]; obs=[]; end


% load list of spectrogram files from step 1
d = expLoad(config, [], 1, 'data');
data = expLoad(config, [], 3, 'data');
% if (strcmp(setting.split, 'train'))
%     d.testPath = d.trainPath;
%     d.testFiles = d.trainFiles;
% end

if exist(d.normFile, 'file')
    specNorm = readNPY(d.normFile);
else
    specNorm = [];
end

idx = 0;
obs.audioFileNames = {};

for k=1:length(d.testFiles)
    sRefMag = readNPY(d.testFiles{k});
    sRefPhase = readNPY(strrep(d.testFiles{k}, '_magnitude.npy', '_phase.npy'));
    
    
    
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
    if (isfield(config, 'debug') && config.debug)
        %         expRandomSeed();
        global iddg
        switch setting.method,
            case 'dnn'
                if config.debug==1
                    iddg = 1150 % randi(size(sRefMag, 1)-100)
                else
                    iddg= config.debug
                end
                
                pred = sPredMag(iddg:iddg+30, :);
                pred(pred<0) = 0;
                pred = mean(pred);
                clf
%       imagesc((flipud(sRefMag(iddg:iddg+10, :)')))
                hold on
                plot(pred)
                legend({'proposed'})
                
            case 'replicate'
                  pred = sPredMag(iddg:iddg+30, :);
                pred(pred<0) = 0;
                pred = mean(pred);
                plot(pred)
                
                 ref = mean(sRefMag(iddg:iddg+30, :));
                 plot(ref, 'k', 'linewidth', 2)
                
                hLegend = findobj(gcf, 'Type', 'Legend');
                legend({hLegend.String{:} setting.method})
                hLegend = findobj(gcf, 'Type', 'Legend');
                     legend({hLegend.String{:} 'reference'})
                xlabel('Frequency (bins)')
                ylabel('Magnitude')
                set(gca, 'fontsize', 16);
                axis tight
                plot([64 64], [0 max(ylim)], 'k:', 'linewidth', 2)
                set(gca,'TickLabelInterpreter','latex')
            otherwise
                pred = sPredMag(iddg:iddg+30, :);
                pred(pred<0) = 0;
                pred = mean(pred);
                plot(pred)
                hLegend = findobj(gcf, 'Type', 'Legend');
                legend({hLegend.String{:} setting.method})
        end
        %         clf
        
        return
    end
end