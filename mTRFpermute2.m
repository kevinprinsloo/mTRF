function [TRF] = mTRFpermute2(stim,resp,fs,map,tmin,tmax,lambda,iter)
%% TO DO
% check all variable dimensions
% optimize, preallocate, etc.
% input check
% test multiple features
% check permuted predictions output
% add flexibility for shuffling at different points?
% normalized model performance? convert r to z or d' using null distribution
% permutation test between models? 


%mTRFpermute mTRF Toolbox cross-validated permutation function.
%   [TRF] = MTRFPERMUTE(STIM,RESP,FS,MAP,TMIN,TMAX,LAMBDA,ITER) performs
%   leave-one-out cross-validation on the set of stimuli STIM and the
%   neural responses RESP and computes  for the range of ridge parameter values LAMBDA.
%   As a measure of performance, it returns the correlation coefficients R
%   between the predicted and original signals, the corresponding p-values
%   P and the mean squared errors MSE. Pass in MAP==1 to map in the forward
%   direction or MAP==-1 to map backwards. The sampling frequency FS should
%   be defined in Hertz and the time lags should be set in milliseconds
%   between TMIN and TMAX.
%
%   [...,PRED,MODEL] = MTRFCROSSVAL(...) also returns the predictions PRED
%   and the linear mapping functions MODEL.
%
%   Inputs:
%   stim   - set of stimuli [cell{1,trials}(time by features)]
%   resp   - set of neural responses [cell{1,trials}(time by channels)]
%   fs     - sampling frequency (Hz)
%   map    - mapping direction (forward==1, backward==-1)
%   tmin   - minimum time lag (ms)
%   tmax   - maximum time lag (ms)
%   lambda - ridge parameter values
%   iter   - if set to > 1, number of permutations for the null distribution
%
%   Output, TRF structure with the following fields:
%   r      - correlation coefficients
%   p      - p-values of the correlations
%   mse    - mean squared errors
%   pred   - prediction [MAP==1: cell{1,trials}(lambdas by time by chans),
%            MAP==-1: cell{1,trials}(lambdas by time by feats)]
%   model  - linear mapping function (MAP==1: trials by lambdas by feats by
%            lags by chans, MAP==-1: trials by lambdas by chans by lags by
%            feats)
%   t      - vector of time lags used (ms)
%   For the shuffled permutations (indicated by trailing '_perm'):
%   r      - null distribution of r values against which TRF.r is tested
%            (lambda by iter)
%   p      - p-values of the permutation test
%   pred   - predicted variable averaged across trials of each permutation
%   model  - linear mapping function averaged across trials of each
%            permutation
%
%   See README for examples of use.
%
%   See also LAGGEN MTRFTRAIN MTRFPREDICT MTRFCROSSVAL MTRFMULTICROSSVAL.

%   References:
%      [1] Crosse MC, Di Liberto GM, Bednar A, Lalor EC (2015) The
%          multivariate temporal response function (mTRF) toolbox: a MATLAB
%          toolbox for relating neural signals to continuous stimuli. Front
%          Hum Neurosci 10:604.
%      [2] Is there a reference for permutaion testing in TRFs?

%   Author: Aaron Nidiffer
%   Lalor Lab, University of Rochester, Rochester, NY, USA
%   Email: edmundlalor@gmail.com
%   Website: http://lalorlab.net/
%   April 2019; Last revision: 5 April 2019
if tmin > tmax
    error('Value of TMIN must be < TMAX')
end
if ~exist('iter','var')
    iter = 1;
end
nlambda = length(lambda);
ntrials = length(stim);

% Convert time lags to samples
tmin_s = floor(tmin/1e3*fs*map);
tmax_s = ceil(tmax/1e3*fs*map);
t = (tmin_s:tmax_s)/fs*1e3;

%% Train all possible combinations
XTX = cell(ntrials,ntrials);
XTy = cell(ntrials,ntrials);

for jj = 1:ntrials
    for kk = 1:ntrials
        
        % Define x and y
        if map == 1
            x = stim{jj};
            y = resp{kk};
        elseif map == -1
            x = resp{kk};
            y = stim{jj};
            [tmin,tmax] = deal(tmax,tmin);
        else
            error('Value of MAP must be 1 (forward) or -1 (backward)')
        end
        
        % Generate lag matrix
        X = [ones(size(x)),lagGen(x,tmin_s:tmax_s)];
        
        % Calculate model
        XTX{jj,kk} = (X'*X);
        XTy{jj,kk} = (X'*y);
        
    end
end

% Set up regularisation
dim = size(X,2);
if size(x,2) == 1
    d = 2*eye(dim,dim);d([1,end]) = 1;
    u = [zeros(dim,1),eye(dim,dim-1)];
    l = [zeros(1,dim);eye(dim-1,dim)];
    M = d-u-l;
else
    M = eye(dim,dim);
end

%% Test matched combinations
for ii = 1:nlambda
    real_XTX=[];
    real_XTy=[];
    for jj = 1:ntrials
        real_XTX(jj,:,:) = XTX{jj,jj};
        real_XTy(jj,:,:) = XTy{jj,jj};
        temp = (XTX{jj,jj}+lambda(ii)*M)\XTy{jj,jj};
        real_model(jj,ii,:,:,:) = reshape(temp(size(x,2)+1:end,:),size(x,2),length(tmin_s:tmax_s),size(y,2));
    end
    
    for jj = 1:ntrials
        trials = 1:ntrials;
        trials(jj) = [];
        
        % Set up Cross-Validation
        current_XTX = shiftdim(sum(real_XTX(trials,:,:),1),1);
        current_XTy = shiftdim(sum(real_XTy(trials,:,:),1),1);
        
        current_model = (current_XTX+lambda(ii)*M)\current_XTy;
        current_c = current_model(1:size(x,2),:);
        current_model = reshape(current_model(size(x,2)+1:end,:),size(x,2),length(tmin_s:tmax_s),size(y,2));
        
        [pred(ii,jj,:,:),r(ii,jj,:),p(ii,jj,:),mse(ii,jj,:)] = mTRFpredict(stim{jj},resp{jj},current_model,fs,map,tmin,tmax,current_c);
    end
end

r_m = mean(r,2);

%% Iteratively test ntrial randomly shuffled combinations
if iter>1
    for pp = 1:iter
        for ii = 1:nlambda
            tr_shuf = randperm(ntrials);
            while any(tr_shuf==1:ntrials)
                tr_shuf = randperm(ntrials);
            end
            shuf_XTX=[];
            shuf_c=[];
            for jj = 1:ntrials
                shuf_XTX(jj,:,:) = XTX{jj,tr_shuf(jj)};
                shuf_XTy(jj,:,:) = XTy{jj,tr_shuf(jj)};
                temp = (XTX{jj,tr_shuf(jj)}+lambda(ii)*M)\XTy{jj,tr_shuf(jj)};
                shuf_model(jj,ii,:,:,:) = reshape(temp(size(x,2)+1:end,:),size(x,2),length(tmin_s:tmax_s),size(y,2));
            end
            model_perm(pp,ii,:,:,:) = mean(shuf_model,1);
            
            for jj = 1:ntrials
                trials = 1:ntrials;
                trials(jj) = [];
                
                % Set up Cross-Validation
                current_XTX = shiftdim(sum(shuf_XTX(trials,:,:),1),1);
                current_XTy = shiftdim(sum(shuf_XTy(trials,:,:),1),1);
                
                current_model = (current_XTX+lambda(ii)*M)\current_XTy;
                current_c = current_model(1:size(x,2),:);
                current_model = reshape(current_model(size(x,2)+1:end,:),size(x,2),length(tmin_s:tmax_s),size(y,2));
                
                [pred_temp(ii,jj,:,:),r_temp(ii,jj,:,:),~,~] = mTRFpredict(stim{jj},resp{jj},current_model,fs,map,tmin,tmax,current_c);
            end
            pred_perm(ii,pp,:,:) = mean(pred_temp,2);
            r_perm(ii,pp,:,:) = squeeze(mean(r_temp,2));
        end
    end
    p_perm = squeeze(1-mean(repmat(r_m,1,size(r_perm,2),1)>r_perm,2));
else
    r_perm = [];
    p_perm = [];
    model_perm = [];
    pred_perm = [];
end

%% Organize output structure
TRF.r = r;
TRF.p = p;
TRF.mse = mse;
TRF.pred = pred;
TRF.model = real_model;
TRF.t = t;

TRF.r_perm = r_perm;
TRF.p_perm = p_perm;
TRF.pred_perm = pred_perm;
TRF.model_perm = model_perm;
