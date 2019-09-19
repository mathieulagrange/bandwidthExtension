function config = baexReport(config)
% baexReport REPORTING of the expLanes experiment bandwithExtension
%    config = baexInitReport(config)
%       config : expLanes configuration state

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

if nargin==0, bandwithExtension('report', 'r'); return; end

config = expExpose(config, 't', 'step', 1, 'mask', {2});
return
% 
mask = {2 2 1 3 2 1 0 1 0 0 2};
config = expExpose(config, 'loss', 'step', 2, 'obs', 0, 'mask', mask, 'pooling', 'l');
return
show=1;

switch show
    case 1 % step 2 loss
        for k=2
            mask = {k 1 1 0 0 0 0 1 0 0 2};
            % config = expExpose(config, 'p', 'step', 2, 'obs', 'lossValidation', 'mask', {k 1 1 2 3 0 0 1 0 0 2}, 'precision', 4, 'expand', 'epochs', 'pooling', 'l', 'percent', 0);
            config = expExpose(config, 'p', 'step', 2, 'obs', 'lossValidation', 'mask', mask, 'precision', 4, 'expand', 'epochs', 'pooling', 'l', 'uncertainty', -1);
            % config = expExpose(config, 'p', 'step', 3, 'obs', 'loss_spec', 'mask', {k 1 1 2 3 0 0 1 0 0 2}, 'precision', 4, 'percent', 0, 'expand', 'epochs');
        end
        
    case 2 % step 4 srr
        for k=1:2
            config = expExpose(config, 'p', 'step', 4, 'obs', [6], 'mask', {k 1 1 2 0 0 0 1 0 0 2}, 'expand', 'epochs', 'negativeRank', [4]);
        end
    case 3
        config = expExpose(config, 't', 'step', 4, 'obs', [6], 'mask', {1 2 0 2 0 0 4 1 0 0 2}, 'negativeRank', [4]);
end
% config = expExpose(config, 't', 'step', 2, 'obs', 0, 'mask', {0 0 1}, 'precision', 4, 'percent', 1);

% config = expExpose(config, 't', 'step', 4, 'obs', 3:5, 'mask', {1 2 [3 4] 3 5 1 0 0 0 0 0 0 0 0 0 0}, 'percent', -1, 'precision', 4, 'negativeRank', [1, 2]);

% config = expExpose(config, 't', 'obs', 1:3, 'mask', {1 split 0 2 3}, 'percent', 0, 'precision', 4);
