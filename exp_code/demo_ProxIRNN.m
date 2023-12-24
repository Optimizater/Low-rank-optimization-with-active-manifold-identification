% test the ProxIRNN / IRNRI / EIRNRI with random data
clc,clear,format long 
rng(22)
nr = 150; nc = 150; r = 15 ;
% --------------- Synthetic data ---------------
xb = 1+randn(nr,r); xc = 2+randn(r,nc) ;
Y = xb*xc ; 
% --------------- random mask ---------------
M_org = zeros(nr,nc); 
missrate = 0.5; 
for i=1:nc  
    idx = 1:1:nr;
    randidx=randperm(nr,nr); % 随机[n] 中的 k 个 index
    M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~M_org; 
Xm=Y.*mask; 

% --------------- parameters ---------------
orieps = 5;
lambda = 1e-5*norm(Y,inf);
% lambda = 1/nr; 
itmax = 1e4; 
%% PIRNN / AIRNN / EPIRNN algorithm
% options: max_iter / eps / mu / Rel / Scalar / alpha
% Sensitivity to the initial value
spl = 0.1 ; % sp norm 
tol_spl = [1e-6; 1e-7; 1e-7]; 
rcSen = 150; 
X0 = (1+randn(nr,rcSen))*randn(rcSen,nc); 
spPIR = {}; spAIR = {}; 
for itsp = 1:length(spl)
  sp = spl(itsp); 
  tol = tol_spl(itsp+1); 
  optionsP.max_iter = itmax; 
  optionsP.Rel = Y; 
  optionsP.eps = orieps; 
  spPIR{itsp} = ProxIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 

  % optionsP.eps = 10 ; 
  optionsP.Scalar = 0.8; 
  spAIR{itsp} = IRNRI(X0,Xm,sp, lambda, mask, tol, optionsP); 

  optionsP.alpha = 5e-1; 
  spEPIR{itsp} = EIRNRI(X0,Xm,sp, lambda, mask, tol, optionsP); 
end
  %%
  Plen = 9e4; PIRx = 1:1:Plen; AIRx = 1:1:Plen; EPIRx = 1:1:Plen;  
  split = 2;
  PIR = spPIR{split}; AIR = spAIR{split}; EPIR = spEPIR{split};
  
  figure(1)
  subplot(2,2,1)
  plot(PIRx,log10(PIR.RelErr(1:Plen)),':.r','linewidth',1);hold on
  plot(AIRx,log10(AIR.RelErr(1:Plen)),'--b','linewidth',1);
  plot(EPIRx,log10(EPIR.RelErr(1:Plen)),'-.g','linewidth',1);hold off
  xlabel("iteration"); ylabel("log(RelErr)")
  legend("PIRNN","AIRNN","EPIRNN")

  subplot(2,2,2)
  plot(PIRx,log10(PIR.RelDist(1:Plen)),':.r','linewidth',1);hold on
  plot(AIRx,log10(AIR.RelDist(1:Plen)),'--b','linewidth',1);
  plot(EPIRx,log10(EPIR.RelDist(1:Plen)),'-.g','linewidth',1);hold off
  xlabel("iteration"); ylabel("log(RelDist)")
  legend("PIRNN","AIRNN","EPIRNN")
  
  subplot(2,2,3)
  plot(PIRx,PIR.f(1:Plen),':.r','linewidth',1);hold on; 
  plot(AIRx,AIR.f(1:Plen),'--b','linewidth',1);
  plot(EPIRx,EPIR.f(1:Plen),'-.g','linewidth',1);hold off
  xlabel("iteration"); ylabel("F(x)")
  legend("PIRNN","AIRNN","EPIRNN")
  
  subplot(2,2,4)
  plot(PIRx,PIR.rank(1:Plen),':.r','linewidth',1);hold on; 
  plot(AIRx,AIR.rank(1:Plen),'--b','linewidth',1);
  plot(EPIRx,EPIR.rank(1:Plen),'-.g','linewidth',1);hold off 
  xlabel("iteration"); ylabel("rank")
  legend("PIRNN","AIRNN","EPIRNN")