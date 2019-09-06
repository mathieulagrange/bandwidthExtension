function config = baexReport(config)                               
% baexReport REPORTING of the expLanes experiment bandwithExtension
%    config = baexInitReport(config)                               
%       config : expLanes configuration state                      
                                                                   
% Copyright: Mathieu Lagrange                                      
% Date: 22-May-2019                                                
                                                                   
if nargin==0, bandwithExtension('report', 'r'); return; end      

mask = {0 0 1 2 3 0 [0] 1 0 0 2};
config = expExpose(config, 'p', 'step', 2, 'obs', 'lossValidation', 'mask', mask, 'precision', 4, 'expand', 'epochs', 'pooling', 'l');

return


for k=1
% config = expExpose(config, 'p', 'step', 2, 'obs', 'lossValidation', 'mask', {k 1 1 2 3 0 0 1 0 0 2}, 'precision', 4, 'expand', 'epochs', 'pooling', 'l', 'percent', 0);
config = expExpose(config, 'p', 'step', 3, 'obs', 'loss_spec', 'mask', {k 1 1 2 3 0 0 1 0 0 2}, 'precision', 4, 'percent', 0, 'expand', 'epochs');

end

% config = expExpose(config, 't', 'step', 2, 'obs', 0, 'mask', {0 0 1}, 'precision', 4, 'percent', 1);

% config = expExpose(config, 't', 'step', 4, 'obs', 3:5, 'mask', {1 2 [3 4] 3 5 1 0 0 0 0 0 0 0 0 0 0}, 'percent', -1, 'precision', 4, 'negativeRank', [1, 2]);

% config = expExpose(config, 't', 'obs', 1:3, 'mask', {1 split 0 2 3}, 'percent', 0, 'precision', 4);                                   
