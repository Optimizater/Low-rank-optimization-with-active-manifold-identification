%% test the PIRNN / AIRNN / EPIRNN with random data
clc,clear,format long
rng(22)
nr = 150; nc = 150; r = 15 ;
% --------------- Synthetic data ---------------
xb = randn(nr,r); xc = randn(r,nc) ;
Y = xb*xc ;
% --------------- random mask ---------------
M_org = zeros(nr,nc);
missrate = 0.5;
for i=1:nc
  idx = 1:1:nr;
  randidx=randperm(nr,nr); % random sequence
  M_org(randidx(1:ceil(nr*missrate)),i)=1;
end
mask = ~M_org; Xm=Y.*mask;
% --------------- parameters ---------------
lambda = 1e-4*norm(Xm,inf);
itmax = 2e4;
sp = 0.1;
tol = 1e-5;
%% basic algorithm for sp=0.5 ,SR=0.5
% Initial point
% with default eps = eps(1)
rcSen = 15; 
X0 = (1+randn(nr,rcSen))*(randn(rcSen,nc));
optionsP.Rel = Y;
optionsP.max_iter = itmax;
PIR = ProxIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);

optionsP.Scalar = 0.8;
AIR = IRNRI(X0,Xm,sp, lambda, mask, tol, optionsP);

optionsP.alpha = 5e-1;
optionsP.eps = eps(1);
EPIR = EIRNRI(X0,Xm,sp, lambda, mask, tol, optionsP);

optionsP.max_iter = 100;
SCP = MC_SCpADMM(X0,Xm,sp, lambda, mask, tol, optionsP);
%%
%% plot
pPIR = min(itmax,PIR.iterTol); pAIR = min(itmax,AIR.iterTol);
pEPI = min(itmax,EPIR.iterTol);
PIRx = (1:1:pPIR); AIRx = (1:1:pAIR); EPIRx = (1:1:pEPI);
% ------------ RelErr plot
h = figure(1);
%     set (gca,'position',[0.1,0.1,0.9,0.9] );
set(h,'Position',[500 500 1500 500]);
%     subplot(1,3,1)
subplot('Position',[0.05,0.1,0.28,0.85])
plot(PIRx,log10(PIR.RelErr(1:pPIR)),':.k','linewidth',1);hold on
plot(AIRx,log10(AIR.RelErr(1:pAIR)),'--b','linewidth',1);
plot(EPIRx,log10(EPIR.RelErr(1:pEPI)),'-.r','linewidth',1); hold off
xlabel("iteration"); ylabel("log(RelErr)")
legend("PIRNN","AIRNN","EPIRNN")
subplot('Position',[0.38,0.1,0.28,0.85])
%     subplot(1,3,2,'position',[0.35,0,0.3,1])
plot(PIRx,log10(PIR.RelDist(1:pPIR)),':.k','linewidth',1);hold on
plot(AIRx,log10(AIR.RelDist(1:pAIR)),'--b','linewidth',1);
plot(EPIRx,log10(EPIR.RelDist(1:pEPI)),'-.r','linewidth',1);hold off
xlabel("iteration"); ylabel("log(RelDist)")
legend("PIRNN","AIRNN","EPIRNN")
subplot('Position',[0.71,0.1,0.28,0.85])
%     subplot(1,3,3,'position',[0.7,0,0.3,1])
plot(PIRx,PIR.f(1:pPIR),':.k','linewidth',1);hold on;
plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',1);
plot(EPIRx,EPIR.f(1:pEPI),'-.r','linewidth',1);hold off
xlabel("iteration"); ylabel("F(x)")
legend("PIRNN","AIRNN","EPIRNN")


%%  SCp-norm
opt.p =0.01; % value of p

EPIR_fro = zeros(100,1); % mse
EPIR_peak_snr = zeros(100,1); % psnr
EPIR_mean_snr = zeros(100,1); % snr
options.max_iter = 100 ;
options.eps = 1;
options.Scalar = 0.1 ;

KLopt = 1e-4 ;
Rank_EPIR = zeros(100,3) ; % save rank

lambda = 1 ;
EPIRNN_sol = cell(100,3);
% lambda = 1e-4*norm(opt.D_omega,inf);
%%
for i =100
  opt.p = 1
  img_new = zeros(size(img_ori)) ;
  options.KLopt = i/10*KLopt ;
  for channel = 1:3
    opt.omega = omega(:,:,channel); % mask
    opt.D_omega = omega_img(:,:,channel); % observation matrix

    Par = EIRNRI(img_new(:,:,channel),opt.D_omega, opt.p, lambda/10, opt.omega, 1e-5, options);
    EPIRN_sol{i,channel} = Par;

    img_new(:,:,channel) = Par.Xsol;
  end
  [peaksnr,snr] = psnr(img_new,img_ori);

  fro_norm = norm(img_ori(:) - img_new(:), 2)/norm(img_ori(:), 2);
  %     fprintf('p: %2.2f, ', opt.p);
  %     fprintf('fro_norm: %f, ', fro_norm);
  %     fprintf('peaksnr: %f, ', peaksnr);
  %     fprintf('snr: %f.\n', snr);

  EPIR_fro(i) = fro_norm;
  EPIR_peak_snr(i) = peaksnr;
  EPIR_mean_snr(i) = snr;
  %   opt.p = 0.01 + opt.p;
end
%%
imshow(img_new)
%% plot F(X)
plot(EPIRN_sol{2,1}.f)
%%
plot(EPIR_peak_snr)
hold on
plot(peak_snr)
legend("EPIR","SCP")
%%
[hAx,hline1,hline2] = plotyy(0.01:0.01:1,EPIR_peak_snr,0.01:0.01:1,EPIR_fro);%,1:100,mean_snr);
ylabel(hAx(1),'PSNR'); % left y-axis
ylabel(hAx(2),'RE'); % right y-axis
legend([hline1,hline2],{'PSNR';'RE'});
%%
[r1,r2] = max(peak_snr);
%%
[mpsnr,index] = max(peak_snr);