

for d=1:2
    if d==1
        db = 'gtzan';
        ids = [1141 279 664 412 1120];
        ids=1120;
    else
        db = 'solos';
        ids = [293 1129 597 342 803 1121 336 546];
        ids= 1141;
    end
    
    for k=1:length(ids)
        bandwidthExtension('do', 5, 'mask', {d+1 2 [1 2 5 6] 3 3 2 3 2 1 1 1 1 1 1 1}, 'debug', ids(k));
        fileName = ['paper/' db '_' num2str(ids(k))];
        export_fig([fileName  '.png'], '-transparent');
        savefig([fileName  '.fig'])
    end
end