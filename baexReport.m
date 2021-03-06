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
show=6;


switch show
    case 1 % step 2 loss
        mask = {2 0 1 4:5 4:5 3 1:3 1 2};
        mask = {1 1 1 3 3 0 0 0 2};
        %         mask = {{2 0 1 4 4:5 3 1:3 2:6 2}};
        %         mask = {2 0 1 3 4:5 3 1:3 5:6 2};
        
        mask = {2, 1, 1, 0, 0, 0, [1  2  3], 0, 0, 1};
        mask = {[2 3] 0 [5 6] 0 0 0 0 0 0 1};
        mask = {{[3], 1, [1 5 6], 3, 2, 2, 0, 2, 0, 1, 0, 0, 0, 0, 1}};
        db = 3;
        mask = {{db, 1, [1], 3, 2, 2, 3, 2, 0, [2 4], 0, 1, 0, 0, 1}, {db, 1, [5 6], 3, 2, 2, 11, 2, 0, [2 4], 0, 1, 0, 0, 1}};
        config = expExpose(config, 't', 'step', 2, 'obs', [2 3], 'mask', mask, 'negativeRank', [1 2 3], 'precision', 4, 'pooling', 'l', 'uncertainty', 0, 'highlight', 0, 'save', 1, 'name', '+');
        
    case 2 % step 2 loss per setting
        clf
        hold on
        leg = {};
        for k=3
            for l=3
                for m=1
                    mask = {2 1 [5 6] k l 3 0 m 0 0};
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
        db = 3;
        mask = {{[2 3], 1, [1], 2:3, 0, 2, 3, 3, 0, [2], 0, 1, 0, 0, 1}};
        mask = {[2  3], 1, 1, [2  3], 0, 2, 3, 3, 0, 2, 0, 1, 0, 0, 1};
        config = expExpose(config, 't', 'step', 2, 'obs', 3, 'mask', mask, 'precision', 3, 'save', 0, 'name', '+', 'negativeRank', [1 2], 'highlight', -1);
    case 5 % odg ops
        db = 3;
        mask = {{[2 3], 1, [1], 2:3, 0, 2, 3, 3, 0, [2], 0, 1, 0, 0, 1}};
        mask = {{[2 3], 1, [1:6], 3, 2, 2, 3, 2, 0, 2, 0, 0, 0, 0, 1}};
        mask = {{[2], 2, [1, 5, 6], 3, 2, 2, [3], 2, 0, [2], 0 , 0, 0, 0, 1}};
        config = expExpose(config, 't', 'step', 4, 'obs', 8, 'mask', mask, 'precision', 1, 'save', 0, 'name', '+', 'negativeRank', [1 2], 'highlight', -1, 'expand', 'dataset');
    case 4
        mask = {{[2 3], 1, [1], 3, 2, 2, 0, 0, 0, 1}};
        db = 3;
        mask = {{db, 1, [1 2 3 4], 3, 2, 2, 3, 2, 0, [2 4], 0, 1, 0, 0, 1}, {db, 1, [5 6], 3, 2, 2, 11, 2, 0, [2 4], 0, 1, 0, 0, 1}};
        config = expExpose(config, 't', 'step', 4, 'obs', 7, 'mask', mask, 'negativeRank', 4:6, 'precision', 3, 'sort', -3, 'save', 1, 'name', '+');
    case 6
             mask = {{[2 3], 1, [0], 3, 2, 2, [3], 2, 0, [2], 0 , 1, 0, 0, 1}};
        config = expExpose(config, 't', 'step', 4, 'obs', 0, 'mask', mask, 'precision', 1, 'percent', 1, 'precision', 3, 'save', 0, 'name', '+', 'negativeRank', [1 2], 'highlight', -1);
   

end
% config = expExpose(config, 't', 'step', 2, 'obs', 0, 'mask', {0 0 1}, 'precision', 4, 'percent', 1);

% config = expExpose(config, 't', 'step', 4, 'obs', 3:5, 'mask', {1 2 [3 4] 3 5 1 0 0 0 0 0 0 0 0 0 0}, 'percent', -1, 'precision', 4, 'negativeRank', [1, 2]);

% config = expExpose(config, 't', 'obs', 1:3, 'mask', {1 split 0 2 3}, 'percent', 0, 'precision', 4);
