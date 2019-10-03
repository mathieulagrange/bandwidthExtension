function config = baexReport(config)
% baexReport REPORTING of the expLanes experiment bandwidthExtension
%    config = baexInitReport(config)
%       config : expLanes configuration state

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

if nargin==0, bandwidthExtension('report', 'rH'); return; end


% mask = {2 2 1 3 2 1 0 1 0 0 2};
% config = expExpose(config, 'loss', 'step', 2, 'obs', 0, 'mask', mask, 'pooling', 'l');
% return
show=4;

switch show
    case 1 % step 2 loss
        mask = {2 0 1 4:5 4:5 3 1:3 1 2};
        mask = {1 1 1 3 3 0 0 0 2};
%         mask = {{2 0 1 4 4:5 3 1:3 2:6 2}};
%         mask = {2 0 1 3 4:5 3 1:3 5:6 2};
        config = expExpose(config, 'p', 'step', 2, 'obs', 'lossValidation', 'mask', mask, 'negativeRank', [1 2 3], 'precision', 4, 'expand', 'epochs', 'pooling', 'l', 'uncertainty', -1, 'highlight', -1);

    case 2 % step 2 loss per setting
        clf
        hold on
        leg = {};
        for k=3:5
            for l=3:5
                for m=1:4
                    mask = {2 1 1 k l 3 1:3 m 0 0 2};
                    config.displayData.plotSuccess = 0;
                    config = expExpose(config, 'loss', 'step', 2, 'obs', 0, 'mask', mask);
                    f = [num2str(config.factors.values{4}{k}) '-' num2str(config.factors.values{5}{l}) '-' num2str(config.factors.values{8}{m})];
                    if config.displayData.plotSuccess == 1
                        leg = {leg{:}  [f '-test']}; % [f '-train']
                    end
                end
            end
        end
        legend(leg)

    case 3 % step  timing
        for k=2
            config = expExpose(config, 't', 'step', 2, 'obs', 'time', 'mask', {k 1 1 3 0 1}, 'expand', 'epochs', 'pooling', 'l');
        end
    case 4
        config = expExpose(config, 't', 'step', 4, 'obs', 4:7, 'mask', {2, 2, 0, 3, 4, 3, 2, 3, 2}, 'negativeRank', 4:6);
end
% config = expExpose(config, 't', 'step', 2, 'obs', 0, 'mask', {0 0 1}, 'precision', 4, 'percent', 1);

% config = expExpose(config, 't', 'step', 4, 'obs', 3:5, 'mask', {1 2 [3 4] 3 5 1 0 0 0 0 0 0 0 0 0 0}, 'percent', -1, 'precision', 4, 'negativeRank', [1, 2]);

% config = expExpose(config, 't', 'obs', 1:3, 'mask', {1 split 0 2 3}, 'percent', 0, 'precision', 4);
