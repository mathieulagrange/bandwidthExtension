function config = baexReport(config)                               
% baexReport REPORTING of the expLanes experiment bandwithExtension
%    config = baexInitReport(config)                               
%       config : expLanes configuration state                      
                                                                   
% Copyright: Mathieu Lagrange                                      
% Date: 22-May-2019                                                
                                                                   
if nargin==0, bandwithExtension('report', 'r'); return; end      

split = 1;

config = expExpose(config, 'loss', 'step', 2, 'obs', 1, 'mask', {2 1 2 3 5}, 'precision', 4, 'save', 0);

return 

config = expExpose(config, 't', 'step', 3, 'obs', 1:3, 'mask', {1 split 2 2 3}, 'percent', 0, 'precision', 4);

config = expExpose(config, 't', 'obs', 1:3, 'mask', {1 split 0 2 3}, 'percent', 0, 'precision', 4);                                   
