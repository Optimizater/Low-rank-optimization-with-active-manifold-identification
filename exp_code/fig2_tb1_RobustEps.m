%% ------------- Robust Experiment -------------
% Rank Robust
% rank from 5 , 10, record the recognized success rank
clc,clear,format long g; rng(22);
% p = parpool(16);
nr = 150; nc = 150;
% ---------------- random mask ----------------
missrate = 0.5;
% %% --------------- parameters ---------------
lambda = 0;
itmax = 1e3;
sp = 0.5;
tol = 1e-7;
klopt = 1e-5;
weps = 1e-16;

options.max_iter = itmax;
options.KLopt = klopt;
options.beta = 1.1;
options.teps = weps;

Rank = [5,10,15];
WEPS = [1e-3]; % 10.^[-1 -2 -3]; % ;
times = 100; % 100
init_rank_start = 1;
init_rank_max = 20;
% number of correct rank
Rank_RC.PIR = zeros(length(Rank),length(WEPS));
Rank_RC.AIR = zeros(size(Rank));
Rank_RC.EPIR = zeros(size(Rank));
% Rank_RC.ACCI = zeros(size(Rank));

% average MSE
AMse.PIR = zeros(length(Rank),length(WEPS));
AMse.AIR = zeros(size(Rank));
AMse.EPIR = zeros(size(Rank));
% AMse.ACCI = zeros(size(Rank));
% Initial point
for irank = 1:length(Rank) %3
  r = Rank(irank);
  parfor r_iter = init_rank_start : init_rank_max %
    par{r_iter} = VsRobustEps(nr,nc,r,r_iter,lambda,sp,missrate,tol,options,WEPS,0,times);
% -------------------------------------
  end
  for iter = init_rank_start : init_rank_max
    Rank_RC.PIR(irank,:) = Rank_RC.PIR(irank,:) + par{iter}.CrRank.PIR;
    Rank_RC.AIR(irank) = Rank_RC.AIR(irank) + par{iter}.CrRank.AIR;
    Rank_RC.EPIR(irank) = Rank_RC.EPIR(irank) + par{iter}.CrRank.EPIR;
%     Rank_RC.ACCI(irank) = Rank_RC.ACCI(irank) + par{iter}.CrRank.ACCI;
    
    AMse.PIR(irank,:) = AMse.PIR(irank,:) + (par{iter}.MSE.PIR)./times;
    AMse.AIR(irank) = AMse.AIR(irank) + (par{iter}.MSE.AIR)./times;
    AMse.EPIR(irank) = AMse.EPIR(irank) + (par{iter}.MSE.EPIR)./times;
    %     AMse.ACCI(irank) = AMse.ACCI(irank) + par{iter}.MSE.ACCI;
  end
  plust = nr*nc*init_rank_max;
  AMse.PIR = (AMse.PIR)./init_rank_max;
  AMse.AIR = (AMse.AIR)./init_rank_max;
  AMse.EPIR = (AMse.EPIR)./init_rank_max; 
end

Robust_Rank515.CR = Rank_RC;
Robust_Rank515.AMse = AMse;
%% plot the table
% save("..\exp_cache\Rank_Robust.mat","Rank_RC",'-mat')
clear all;clc;
load("..\exp_cache\Rank_Robust.mat","Rank_RC",'-mat')
X = categorical({'EIRNAMI', 'IRNAMI', 'PIRNN'});
% X = reordercats(X,{'Medium','Extra Large'});
for irankplt = 1 : 3
  figure(irankplt)
  bar(X,[Rank_RC.EPIR(irankplt), Rank_RC.AIR(irankplt),...
    Rank_RC.PIR(irankplt,3)])
  ylabel('the number of correct rank')
end

%%

% % -------------------------------------------------------------------% %
%% ----------  percentage of success ----------
% AdaIRNN V.S. ProxIRNN with random data 
% this experiment shows the AdaIRNN are more robust than ProxIRNN
% the AdaIRNN has the similar convergence rate with ProxIRNN if the 
% initialization eps are similar
% --------------- generate data ---------------
clc,clear,format long; rng(23); 
p = parpool(16);
nr = 150; nc = 150; 
% %% --------------- parameters ---------------
lambda = 5e-2;
itmax = 1e3;
sp = 0.5;
tol = 1e-7;
klopt = 1e-5;
beta = 1.1;

success = 1e-2;
% ------------------------------------------------------------------------
missrate = 0.5; 
weps = 1e-4;

Rank = [5:1:10];
init_rank_max = max(nr,nc);
ITRANK = [1:1:20,131:150];
% WEPS = [1e-1, 7e-2, 4e-2, 1e-2];
% WEPS = [5e-1];
WEPS = [7e-1,4e-1,1e-1,7e-2,4e-2,1e-2,7e-3,4e-3,1e-3];
times = 5;

options.max_iter = itmax;
options.KLopt = klopt;
%   options.eps = weps;
options.beta = beta;
% -------------------------- 75 *20 times --------------------------
% with different initialization rank: 0--74
% for each initialization rank we test 20 times
Robust.PIR = zeros(length(Rank),length(WEPS));
Robust.AIR = zeros(size(Rank));
Robust.EPIR = zeros(size(Rank));
CrRank.PIR =zeros(length(Rank),length(WEPS));
CrRank.AIR = zeros(size(Rank));
CrRank.EPIR = zeros(size(Rank));


for irank = 1 : length(Rank) %3
  r = Rank(irank);
  parfor r_iter = 1:length(ITRANK) % 74
    par{r_iter} = VsRobustEps(nr,nc,r,ITRANK(r_iter),lambda,sp,missrate,tol,options,WEPS,[1,success],times);
% -------------------------------------
  end
  for iter = 1:1:length(ITRANK)
    Robust.PIR(irank,:) = Robust.PIR(irank,:) + par{iter}.Robust.PIR;
    Robust.AIR(irank) = Robust.AIR(irank) + par{iter}.Robust.AIR;
    Robust.EPIR(irank) = Robust.EPIR(irank) + par{iter}.Robust.EPIR;
    CrRank.PIR(irank,:) = CrRank.PIR(irank,:) + par{iter}.CrRank.PIR;
    CrRank.AIR(irank) = CrRank.AIR(irank) + par{iter}.CrRank.AIR;
    CrRank.EPIR(irank) = CrRank.EPIR(irank) + par{iter}.CrRank.EPIR;
  end
end

%%
% save(Robust_Eps.mat,Robust,'-mat')
R_CROBIST.Robust = Robust;
R_CROBIST.CrRank = CrRank;    
save("..\exp_cache\R_CROBIST_plot.mat","R_CROBIST",'-mat')
%% 
rck = R_CROBIST.CrRank;
rcs = R_CROBIST.Robust;
plot([5:1:10],rck.PIR(:,1)./200); hold on;
plot([5:1:10],rck.PIR(:,3)./200); 
plot([5:1:10],rck.AIR./200)
plot([5:1:10],rck.EPIR./200)

%%
delete(p);