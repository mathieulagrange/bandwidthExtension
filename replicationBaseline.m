function hf = replicationBaseline(spec, corr)

if (~exist('corr', 'var')), corr=false;end

bf = spec(1:ceil(end/2)-1, :);
hfr =  spec(ceil(end/2)+1:end, :);

hf = bf/sum(bf(:))*sum(hfr(:));

if (corr)
    [c lags] = xcorr(mean(hf, 2), mean(hfr, 2));
    [~, ind] =  max(c);
    hf = circshift(hf, -ind);
end

ls = filter(fir1(20, .25), 1, [bf; hf]);
w = hanning(20);
for k=1:10
    hf(k, :) = w(k)*hf(k, :)+w(10+k)*ls(k+ceil(size(spec, 1)/2)+1, :);
    if (corr)
        hf(end+1-k, :) = w(k)*hf(end+1-k, :);
    end
end
