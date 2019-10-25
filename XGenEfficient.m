function X = XGenEfficient(x,lags)
%lagGen Lag generator.
%   [XLAG] = LAGGEN(X,LAGS) returns the matrix XLAG containing the lagged
%   time series of X for a range of time lags given by the vector LAGS. If
%   X is multivariate, LAGGEN will concatenate the features for each lag
%   along the columns of XLAG.
%
%   Inputs:
%   x    - vector or matrix of time series data (time by features)
%   lags - vector of integer time lags (samples)
%
%   Outputs:
%   xLag - matrix of lagged time series data (time by lags*feats)
%
%   See README for examples of use.
%
%   See also MTRFTRAIN MTRFPREDICT MTRFCROSSVAL MTRFMULTICROSSVAL.

%   Author: Michael Crosse
%   Lalor Lab, Trinity College Dublin, IRELAND
%   Email: edmundlalor@gmail.com
%   Website: http://lalorlab.net/
%   April 2014; Last revision: 18 August 2015

% xLag = zeros(size(x,1),size(x,2)*length(lags));
X = zeros(size(x,1),size(x,2)+size(x,2)*length(lags));

X(:,1:size(x,2)) = ones(size(x));

i = 1;
for j = 1:length(lags)
    if lags(j) < 0
        X(1:end+lags(j),size(x,2)+i:size(x,2)+i+size(x,2)-1) = x(-lags(j)+1:end,:);
    elseif lags(j) > 0
        X(lags(j)+1:end,size(x,2)+i:size(x,2)+i+size(x,2)-1) = x(1:end-lags(j),:);
    else
        X(:,size(x,2)+i:size(x,2)+i+size(x,2)-1) = x;
    end
    i = i+size(x,2);
end

end