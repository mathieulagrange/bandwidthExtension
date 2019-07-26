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
if nargin==0, bandwithExtension('do', 4, 'mask', {1 1 2 3 4 1 1}); return; else store=[]; obs=[]; end

% load list of spectrogram files from step 1
d = expLoad(config, [], 1, 'data');
reference=[];
for k=1:length(d.testFiles)
    rk = readNPY(d.testFiles{k});
    reference((k-1)*500+1:(k-1)*500+size(rk, 1), :, :) = rk;
end

% mel projection matrices
mel27 = fft2melmx(setting.frameSize, setting.samplingFrequency, 27);
mel27 = mel27(:, 1:end/2+1);
mel40 = fft2melmx(setting.frameSize, setting.samplingFrequency, 40);
mel40 = mel40(:, 1:end/2+1);

for k=1:size(reference, 1)
    refk = squeeze(reference(k, :, :));
    switch setting.method
        case 'dnn'
            hf = squeeze(data.predictions(k, :, :))';
        case 'replication'
            hf = replicationBaseline(refk);
        case 'replicationCorr'
            hf = replicationBaseline(refk, true);
        case 'oracle'
            hf = refk(ceil(end/2)+1:end, :);
        case 'null'
            hf = zeros(ceil(size(refk, 1)/2)-1, size(refk, 2));
    end
    pred = refk;
    pred(ceil(end/2)+1:end, :) = hf;
%     imagesc(([refk pred]))
    plot([mean(refk, 2) mean(pred, 2)])
    legend({'reference', 'prediction'})
    lossSpec(k) = immse(refk, pred);
    lossCqt27(k) = immse(mel27*refk.^2, mel27*pred.^2);
    lossCqt40(k) = immse(mel40*refk.^2, mel40*pred.^2);
end

obs.lossSpec = lossSpec;
obs.lossCqt27 = lossCqt27;
obs.lossCqt40 = lossCqt40;