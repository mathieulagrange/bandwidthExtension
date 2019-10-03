function [config, store] = baexInit(config)
% baexInit INITIALIZATION of the expLanes experiment bandwidthExtension
%    [config, store] = baexInit(config)
%      - config : expLanes configuration state
%      -- store  : processing data to be saved for the other steps

% Copyright: Mathieu Lagrange
% Date: 22-May-2019

if nargin==0, bandwidthExtension(); return; else store=[];  end
