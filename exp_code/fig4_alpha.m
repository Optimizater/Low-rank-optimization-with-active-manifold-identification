%% -------------------- sensitive of alpha --------------------
% sensitive of alpha for extrapolation parameter 
clc,clear,format long; rng(23)
nr = 150; nc = 150; r = 15;
% --------------- Synthetic data ---------------
Y = randn(nr,r) * randn(r,nc);
% --------------- random mask ---------------
M_org = zeros(nr,nc); missrate = 0.5; 
for i=1:nc 
  randidx=randperm(nr,nr); % random sequence
  M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end

mask = ~M_org; Xm = Y.*mask;

% %% --------------- parameters ---------------
lambda = 1e-1*norm(Xm,inf);
itmax = 1e3; 
sp = 0.5; 
tol = 1e-7; 
klopt = 1e-5;
seps = 1e-3;
% weps = 1e-16;
options.Rel = Y; 
options.max_iter = itmax; 
options.KLopt = klopt;
options.eps = seps; 
options.beta = 1.1; 


optionsEP = options;
Lalpha = [0 0.1 0.3 0.5 0.7 0.9];
for i = 1:length(Lalpha)
  optionsEP.alpha = Lalpha(i) ;
  LEPIR{i} = EIRNRI(Xm,Xm,sp, lambda, mask, tol, optionsEP);
end
%% plot 
%    set(h,'Position',[500 500 1500 500]);
% figure(1) % RelErr
% set (gcf,'position',[150 50 600 400] );
% % set (gca,'position',[0.08,0.1,0.9,0.9]);
% % set(gca,'looseInset',[0 0 0 0])
%   pltLEPx = min(itmax,LEPIR{i}.iterTol);
%   plot(log10(LEPIR{1}.RelErr),':dg','linewidth',1);hold on
%   plot(log10(LEPIR{2}.RelErr),'--b','linewidth',2);
%   plot(log10(LEPIR{3}.RelErr),'-m','linewidth',2);
%   plot(log10(LEPIR{4}.RelErr),'-.k','linewidth',2);
%   plot(log10(LEPIR{5}.RelErr),':>c','linewidth',1);
%   plot(log10(LEPIR{6}.RelErr),'-r','linewidth',2);hold off
%   xlabel("iteration"); ylabel("log_{10}(RelErr)")
%   legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
%     "\alpha = 0.7","\alpha = 0.9")
% % set (gca,'fontsize',12);

figure(2) % RelDist
set (gcf,'position',[50 50 600 600] );
% set (gca,'position',[0.08,0.08,0.9,0.9] );
  plot(log10(LEPIR{1}.RelDist),':dg','linewidth',1);hold on
  plot(log10(LEPIR{2}.RelDist),'--b','linewidth',2);
  plot(log10(LEPIR{3}.RelDist),'-m','linewidth',2);
  plot(log10(LEPIR{4}.RelDist),'-.k','linewidth',2);
  plot(log10(LEPIR{5}.RelDist),':>c','linewidth',1);
  plot(log10(LEPIR{6}.RelDist),'-r','linewidth',2);hold off
  xlabel("iteration"); ylabel("log_{10}(RelDist)")
  legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
  "\alpha = 0.7","\alpha = 0.9")
% set (gca,'fontsize',18);

% figure(3) % obj
% set (gcf,'position',[100 10 600 400] );
% % set (gca,'position',[0.08,0.08,0.9,0.9] );
%   plot(log10(LEPIR{1}.f),':dg','linewidth',1);hold on
%   plot(log10(LEPIR{2}.f),'--b','linewidth',2);
%   plot(log10(LEPIR{3}.f),'-m','linewidth',2);
%   plot(log10(LEPIR{4}.f),'-.k','linewidth',2);
%   plot(log10(LEPIR{5}.f),':>c','linewidth',1);
%   plot(log10(LEPIR{6}.f),'-r','linewidth',1);hold off
%   xlabel("iteration"); ylabel("F(X)")
%   legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
%   "\alpha = 0.7","\alpha = 0.9")
% set (gca,'fontsize',18);
% -/end -------------------------------------------------------------------
