function config = baexReport(config)
% baexReport REPORTING of the expLanes experiment bandwithExtension
%    config = baexInitReport(config)
%       config : expLanes configuration state

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

if nargin==0, bandwithExtension('report', 'rh'); return; end


% mask = {2 2 1 3 2 1 0 1 0 0 2};
% config = expExpose(config, 'loss', 'step', 2, 'obs', 0, 'mask', mask, 'pooling', 'l');
% return
show=4;

switch show
    case 1 % step 2 loss
        for k=2
            mask = {2, 1, 1, 3, 0, 1, 0, 0, 0, 0, 2};
            config = expExpose(config, 'p', 'step', 4, 'obs', 'time', 'mask', mask, 'precision', 4, 'expand', 'epochs', 'pooling', 'l', 'uncertainty', -1);
        end
        
    case 2 % step 2 loss per setting
        clf
        hold on
        leg = {};
        for k=3
            for l=1:3
                for m=1
                    mask = {2 1 1 k l m 1:6 1 0 0 2};
                    config = expExpose(config, 'loss', 'step', 2, 'obs', 0, 'mask', mask);
                    f = [num2str(config.factors.values{4}{k}) '-' num2str(config.factors.values{5}{l}) '-' num2str(config.factors.values{6}{m})];
                    leg = {leg{:} [f '-train'] [f '-test']};
                end
            end
        end
        legend(leg)
        
    case 3 % step 4 srr
        for k=1:2
            config = expExpose(config, 't', 'step', 4, 'obs', 'time', 'mask', {k 1 1 0 0 0 0 1 0 0 2}, 'negativeRank', [4]);
        end
    case 4
        config = expExpose(config, 't', 'step', 4, 'obs', 4:7, 'mask', {2, 2, 1:4, 3, 0, 1, 6}, 'negativeRank', 4:6);
end
% config = expExpose(config, 't', 'step', 2, 'obs', 0, 'mask', {0 0 1}, 'precision', 4, 'percent', 1);

% config = expExpose(config, 't', 'step', 4, 'obs', 3:5, 'mask', {1 2 [3 4] 3 5 1 0 0 0 0 0 0 0 0 0 0}, 'percent', -1, 'precision', 4, 'negativeRank', [1, 2]);

% config = expExpose(config, 't', 'obs', 1:3, 'mask', {1 split 0 2 3}, 'percent', 0, 'precision', 4);
