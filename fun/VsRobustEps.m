% Written by Ye Wang 03/2022(E-mail: wangye@shanghaitech.edu.cn)
% INPUT:
% - nr,nc,r: matrix nr*nc with rank r
% - sr: random start with rank sr
% - lambda, sp, missrate: model parameters
% - tol,options: glgorithm parameters
% - WEPS: iter_eps for PIRNN
% - SUCC: 
%       : [0 ~] SUCC(1)==0, don't check
%       : [1, tol] SUCC(1)==1, use SUCC(2) to check the relative error 
% - times: repeat for experiments
function Par = VsRobustEps(nr,nc,r,sr,lambda,sp,missrate, ...
  tol,options,WEPS,SUCC,times)
leps = length(WEPS);
Robust.PIR = zeros(1,leps);
Robust.AIR = 0;
Robust.EPIR = 0;
Robust.ACCI = 0;

CrRank.PIR = zeros(1,leps);
CrRank.AIR = 0;
CrRank.EPIR = 0;
CrRank.ACCI = 0;

MSE.PIR = zeros(1,leps);
MSE.AIR = 0;
MSE.ACCI = 0;
MSE.EPIR = 0;

if SUCC(1) == 1
  success = SUCC(2);
else
  success = 0;
end

for itimes = 1 : times %
  if success ==0
    Y = randn(nr,r) * randn(r,nc);
  else
    B = rand(nr,r); C = rand(r,nc); Y = B * C;
    Y = Y./(10*min(min(Y)));
    %       Y = Y./(10*max(max(Y)));
  end
  % --------------- random mask ---------------
  M_org = zeros(nr,nc);
  for i = 1 : nc
    randidx = randperm(nr,nr); % random sequence
    M_org(randidx(1:ceil(nr*missrate)),i)=1;
  end
  mask = ~M_org; Xm = Y.*mask;
  X0 = rand(nr,nc); [uY,sY,vY] = svd(X0);
  X0 = uY(:,1:sr) * sY(1:sr,1:sr) * vY(:,1:sr)';

  if success==0
    lambda = 1e-1 * norm(Xm,inf); % rank recognition
  else
    lambda = 1e-3 * norm(Xm,inf); % succfully recovery
  end

  options.Rel = Y;
  optionsP= options;
  %% ProxIRNN 2017
  for iter_eps = 1:leps
    optionsP.eps = WEPS(iter_eps);
    PIR = ProxIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
    %     if (PIR.rank(end) == r) && (PIR.RelErr(end) <= success)
    if PIR.rank(end) == r
      CrRank.PIR(iter_eps) = CrRank.PIR(iter_eps) + 1;
    end
    if (PIR.RelErr(end) <= success) && (PIR.rank(end) == r)
      Robust.PIR(iter_eps) = Robust.PIR(iter_eps) + 1;
    end
    MSE.PIR(iter_eps) = MSE.PIR(iter_eps) + norm(PIR.Xsol - Y,'Fro');
  end
  %% ACCIRNN_2021
  %   ACCIR = AIRNN_2021(X0,Xm,sp,lambda,mask,tol,options);
  %   if rank(ACCIR.Xsol) == r
  %     CrRank.ACCI = CrRank.ACCI + 1;
  %   end
  %
  %   if (norm(ACCIR.Xsol-Y,'fro')/norm(Y,'fro') <= success) && (AIR.rank(end) == r)
  %     Robust.AIR = Robust.AIR + 1;
  %   end
  %   MSE.ACCI = MSE.ACCI + norm(ACCIR.Xsol - Y,'Fro');

  %% IRNRI
  optionsA = options;
  optionsA.eps = 1e0;
  optionsA.mu = 0.7;
  AIR = IRNRI(X0,Xm,sp, lambda, mask, tol, optionsA);
  %   if (AIR.rank(end) == r) && (AIR.RelErr(end) <= success)

  if AIR.rank(end) == r
    CrRank.AIR = CrRank.AIR + 1;
  end

  if (AIR.RelErr(end) <= success) && (AIR.rank(end) == r)
    Robust.AIR = Robust.AIR + 1;
  end
  MSE.AIR = MSE.AIR + norm(AIR.Xsol - Y,'Fro');
  %% EIRNRI
  optionsEP = optionsA;
  optionsEP.alpha = 7e-1;
  EPIR = EIRNRI(X0,Xm,sp, lambda, mask, tol, optionsEP);
  %   if (EPIR.rank(end) == r) && (EPIR.RelErr(end) <= success)

  if EPIR.rank(end) == r
    CrRank.EPIR = CrRank.EPIR + 1;
  end
  if (EPIR.RelErr(end) <= success) && (EPIR.rank(end) == r)
    Robust.EPIR = Robust.EPIR + 1;
  end
  MSE.EPIR = MSE.EPIR + norm(EPIR.Xsol - Y,'Fro');
  %% save
  Par.Robust = Robust;
  Par.CrRank = CrRank;
  Par.MSE = MSE;
end