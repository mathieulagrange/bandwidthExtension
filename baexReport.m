function config = baexReport(config)                               
% baexReport REPORTING of the expLanes experiment bandwithExtension
%    config = baexInitReport(config)                               
%       config : expLanes configuration state                      
                                                                   
% Copyright: Mathieu Lagrange                                      
% Date: 22-May-2019                                                
                                                                   
if nargin==0, bandwithExtension('report', 'r'); return; end      

% config = expExpose(config, 'loss', 'step', 2, 'obs', 1, 'mask', {1 1 2 3}, 'precision', 4, 'save', 0);

config = expExpose(config, 't', 'step', 4, 'obs', 3:5, 'mask', {1 1 0 3 5 1}, 'percent', -1, 'precision', 4, 'negativeRank', [1, 2]);

% config = expExpose(config, 't', 'obs', 1:3, 'mask', {1 split 0 2 3}, 'percent', 0, 'precision', 4);                                   
