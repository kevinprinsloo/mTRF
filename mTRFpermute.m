function [TRF] = mTRFpermute(stim,resp,fs,map,tmin,tmax,lambda,iter)
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

%% Train all possible combinations
MODEL = cell(nlambda,ntrials,ntrials);
C = cell(nlambda,ntrials,ntrials);
for ii = 1:nlambda
    for jj = 1:ntrials
        for kk = 1:ntrials
            [MODEL{ii,jj,kk}, T, C{ii,jj,kk}] = mTRFtrain(stim{jj},resp{kk},fs,map,tmin,tmax,lambda(ii));
        end
    end
end

%% Test matched combinations
for ii = 1:nlambda
    real_model=[];
    real_c=[];
    for jj = 1:ntrials
        real_model(jj,ii,:,:,:) = MODEL{ii,jj,jj};
        real_c(jj,:,:) = C{ii,jj,jj};
    end
    
    for jj = 1:ntrials
        trials = 1:ntrials;
        trials(jj) = [];
        current_model = shiftdim(mean(real_model(trials,ii,:,:,:),1),2);
        current_c = shiftdim(mean(real_c(trials,:,:),1),1);
        [pred(jj,ii,:,:),r(jj,ii,:),p(jj,ii,:),mse(jj,ii,:)] = mTRFpredict(stim{jj},resp{jj},current_model,fs,map,tmin,tmax,current_c);
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
            
            shuf_model=[];
            shuf_c=[];
            for jj = 1:ntrials
                shuf_model(jj,:,:,:) = MODEL{ii,jj,tr_shuf(jj)};
                shuf_c(jj,:,:) = C{ii,jj,tr_shuf(jj)};
            end
            model_perm(pp,ii,:,:,:) = mean(shuf_model,1);
            
            for jj = 1:ntrials
                trials = 1:ntrials;
                trials(jj) = [];
                current_model = shiftdim(mean(shuf_model(trials,:,:,:),1),1);
                current_c = shiftdim(mean(shuf_c(trials,:,:),1),1);
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
TRF.t = T;

TRF.r_perm = r_perm;
TRF.p_perm = p_perm;
TRF.pred_perm = pred_perm;
TRF.model_perm = model_perm;