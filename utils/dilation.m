nbLayers = 4;

d=3;
k = 13;

clear r
r(1) = k
for l=1:nbLayers-1
   r(l+1) = r(l)+d*(k-1)
end