%% search lambda for LENA.jpg
clear;clc;format long
start = 1;

if start==1
  start=0;
  rng(22);format long
  cwd = fileparts(pwd) ;
  path_lena = strcat(cwd,'\img_image\lena.png');
  path_re1 = strcat(cwd,'\img_image\re1.jpg');
  img_ori = double(imread(path_lena))/255 ;
  % img_ori = double(imread(path_lena))/255;
  img_size = size(img_ori);


  %% mask
  %% random mask
  SR = 0.5; % sampleRate = 1 - missRate
  mask = zeros(img_size(1:2));
  for i=1:img_size(2)
    idx = 1:1:img_size(1) ;
    randidx = randperm(img_size(1),img_size(1)); % 随机[n] 中的 k 个 index
    mask(randidx(1:ceil(img_size(1)*SR)),i)=1;
  end
  % mask should obtain (1-missrate)*m*n elements from the original image

  %%
  %% strictly low rank
  % save the top 20% of the singular value
  rt = ceil(min(size(img_ori(:,:,1)))/5);
  for i=1:3
    [U,S,V]=svd(img_ori(:,:,i));
    Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end

  %% ------------------------ RECOVERY ------------------------

  XM = mask.*Xt;
  tol = 1e-4 ;
  mu = 1.3;
  maxIter = 2e3;
  options.max_iter = maxIter;
  options.mu = mu;
  options.KLopt = tol;
  sp = 0.5;

  %%
  % iterLambdaEPIR = [1e-5:5e-6:1e-4];
  % iterLambdaFGSRp = [1e0:1e-2:1.18];
  iterLambdaEPIR = [6.500000000000001e-05];
  iterLambdaFGSRp = [1.180000000000000];

  for LS_idx=1:length(iterLambdaEPIR)
    %% EIRNRI
    optionsEP = options; optionsEP.eps=1e1;
    optionsEP.Scalar = 0.1; optionsEP.alpha = 0.75;
    for channel=1:1
      Xm = XM(:,:,channel);
      lambda = norm(Xm,"fro")*iterLambdaEPIR(LS_idx);
      EPIR = EIRNRI(Xm,Xm,sp, lambda, mask, tol, optionsEP);
      imgR.epir(:,:,channel) = EPIR.Xsol;
      timeTotal.epir{channel} = EPIR.time;
      Objective.epir{channel} = EPIR.f;
      iterRank.epir{channel} = EPIR.rank;
    end
    disp("---------------------------------- EPIRNN")

    %% FGSR
    optionsFGSR.tol=tol;
    optionsFGSR.maxiter = maxIter*1e1;
    optionsFGSR.p = sp;
    for channel=1:1
      Xm = XM(:,:,channel);
      optionsFGSR.lambda = norm(Xm,"fro")*iterLambdaFGSRp(LS_idx);
      Sol_FGSRP = MC_FGSRp_PALM(Xm,mask,optionsFGSR);
      imgR.fgsrp(:,:,channel) = Sol_FGSRP.Xsol;
      timeTotal.fgsrp{channel} = Sol_FGSRP.time;
      iterRank.fgsrp{channel} = Sol_FGSRP.rank;
    end
    disp("---------------------------------- FGSR")

    LAMpsnr(LS_idx,1) = psnr(img_ori(:,:,1),imgR.epir);
    LAMpsnr(LS_idx,2) = psnr(img_ori(:,:,1),imgR.fgsrp);

  end
end
%%
[~,BLMidxEpir] = max(LAMpsnr(:,1)) ;
[~,BLMidxFgsrp] = max(LAMpsnr(:,2)) ;