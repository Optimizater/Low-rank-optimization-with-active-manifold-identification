%% the different method for solving the matric recovery
% step-1 search best lambda with same p for different method
% step-2 compare different restore pic with PSNR

clear; clc; format long; rng(24);
filename = mfilename('fullpath');
[path, scriptname, ~] = fileparts(filename);
cd(path);
addpath(genpath("../fun"))
addpath(genpath("../compare"))
% addpath(genpath("../AIRNN_2021_ori"))
cwd = fileparts(path);


name_pic = 'sist_gym';
name_suffix = '.jpg';
path_img = strcat(cwd,'\img_image\', name_pic, name_suffix );    % % select the pic
img_ori = double(imread(path_img))/255 ;        
it_mask = 1;                                    % % mask for different methods

%% To search a better parameters.
scan.p = 1;
scan.lambda = 1;
if isempty(gcp('nocreate'))
%   parpool();
end

img_size = size(img_ori);
noise = 1e0*randn(img_size(1:2));

% ---------- strictly low rank ----------
RT = [15,20,25,30,35,40];
% for iter_rt = 1:1
for iter_rt = 1:length(RT)
  rt = RT(iter_rt);
  for i = 1:3
    [U,S,V] = svd(img_ori(:,:,i));
    Xt(:,:,i) = U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end
  %% ---------- mask ----------
  if exist('it_mask','var')==0 || it_mask == 1
    % random mask
    missrate = 0.2; % sampleRate = 1 - missRate
    mask = zeros(img_size(1:2));
    for i=1:img_size(2)
      idx = 1:1:img_size(1) ;
      randidx = randperm(img_size(1),img_size(1)); % 随机[n] 中的 k 个 index
      mask(randidx(1:ceil(img_size(1)*missrate)),i)=1;
    end
    mask = ~mask;
  elseif it_mask == 2
    % block_column mask
    mask_path = strcat(cwd,'\img_mask\block_column.bmp');
    omega = double(imread(mask_path))/255.0;
    omega = imresize(omega,img_size(1:2));
    mask = omega(:,:,1);
  elseif it_mask == 3
    % EN mask
    mask_path = strcat(cwd,'\img_mask\block_square.bmp');
    omega = double(imread(mask_path))/255.0;
    omega = imresize(omega,img_size(1:2));
    mask = omega(:,:,1);
  end
  %% ------------------------ RECOVERY ------------------------
  % start point + extrapolation coefficient + parameters
  XM = mask.*Xt;
  X_INIT_0 = zeros(img_size(1:2)) ;
  X_INIT_RAND1 = (1+randn(img_size(1),rt))*(randn(rt,img_size(2)));
  tol = 1e-5 ;
  max_iter = 1e3;
  Klopt = 1e-7;
  options.max_iter = max_iter;
  options.mu = 1.1;

  optionsEP = options;
  optionsEP.eps = 1;

  optionsScp.max_iter = 5e2;
  optionsFGSR.regul_B = "L2";
  %   options.KLopt = 1e-5;
  %% ------------------% search lambda -----------------------------
  % search with fixed sp
  sp = 0.5;
  scan.deep = 2;
  if exist('scan','var') && isfield(scan,'lambda') && scan.lambda == 1
    Lambda_SCP = reshape([10.^(-4:0.5:1); 2*(10.^(-4:0.5:1))], [], 1);
    [LAMBDA_SCP, SP_SCP] = SearchPara(img_ori(:,:,1), XM(:,:,1), XM(:,:,1), 'SCP', Lambda_SCP, sp, mask, tol, optionsScp);
    lambda_scp = LAMBDA_SCP.opt_para;

%     Lambda_IR = reshape([10.^(-4:1:1); 5*10.^(-4:1:1)], [], 1); % 11.4 : 0.05 : 11.8; % 2.^-(-4:0.1:-3);
    Lambda_IR = reshape([10.^(-0.9 : 0.4 : 1.3); 1.8*10.^(-0.9 : 0.4 : 1.3)], [], 1); %1 : 5 : 30;
    [LAMBDA_IR, SP_IR] = SearchPara(img_ori(:,:,1), XM(:,:,1), XM(:,:,1), 'IR', Lambda_IR, sp, mask, tol, optionsEP);
    lambda_ir = LAMBDA_IR.opt_para;

    Lambda_FGSR = reshape([10.^(-4:1:1); 5*10.^(-4:1:1)], [], 1); % 2.^(-3:1:30);
    [LAMBDA_FGSR, SP_FGSR] = SearchPara(img_ori(:,:,1), XM(:,:,1), XM(:,:,1), 'FGSR', Lambda_FGSR, sp, mask, tol, optionsFGSR);
    lambda_fgsr = LAMBDA_FGSR.opt_para;
  else
    lambda_ir = 0.5;
    lambda_scp = lambda_ir;
    lambda_fgsr = lambda_ir;
  end
  %% ------------------ search p-----------------------------
  % search with fixed lambda
  if exist('scan','var') && isfield(scan,'p') && scan.p == 1
    optionsEP.alpha = 0.7;

    op = 0.01 : 0.03 : 1; % % [0,1] = 0.01, 0.04, ... , 0.01 + 0.03*pidx

    [~, SP_SCP] = SearchPara(img_ori(:,:,1), XM(:,:,1), XM(:,:,1), 'SCP', lambda_scp, op, mask, tol, optionsScp);
    sp_scp = SP_SCP.opt_para;

    [~, SP_IR] = SearchPara(img_ori(:,:,1), XM(:,:,1), XM(:,:,1), 'IR', lambda_ir, op, mask, tol, optionsEP);
    sp_ir = SP_IR.opt_para;

    [~, SP_FGSR] = SearchPara(img_ori(:,:,1), XM(:,:,1), XM(:,:,1), 'FGSR', lambda_fgsr, op, mask, tol, optionsFGSR);
    sp_fgsr = SP_FGSR.opt_para;
  else
    sp_ir = 0.5;
    sp_scp = sp_ir;
    sp_fgsr = sp_ir;
  end

  %% ---------- parameter for different model ----------
  % PIRNN
  optionsP = options;

  % AIRNN
  optionsA = optionsP;
  optionsA.mu = 0.7;
  optionsA.eps = 5e-1;

  % IRNRI
  optionsEP = optionsA;
  optionsEP.alpha = 0.7;
  %% ---------- recovery by different method ----------
  img_show.ori_img = img_ori;
  img_show.low_img = Xt;
  img_show.mask_img = XM;
  Parsol = {};
  % % ---------- PIRNN ---------- 
  for i =1:3
    Xm = XM(:,:,i);
    PIR = ProxIRNN(Xm,Xm,sp_ir, lambda_ir, mask, tol, optionsP);
    X_PIR(:,:,i) = PIR.Xsol;
  end
  Parsol{1} = X_PIR;

  % % IRNRI
  %   for i=1:3
  %     Xm = XM(:,:,i);
  %     AIR = IRNRI(Xm+noise,Xm,sp_ir, lambda_ir, mask, tol, optionsA);
  %     X_AIR(:,:,i) = AIR.Xsol;
  %   end
  %   Parsol{2} = X_AIR;

  % % ---------- AccIRNN_2021 ---------- 
  for i=1:3
    Xm = XM(:,:,i);
%     options.Rel = img_ori(:,:,i);
    AIR = AIRNN_2021(Xm,Xm,sp_ir, lambda_ir, mask, tol, options);
    X_AIR(:,:,i) = AIR.Xsol;
  end
  Parsol{2} = X_AIR;

  % % ---------- IRNRI ----------
  for i=1:3
    Xm = XM(:,:,i);
    EPIR = IRNRI(Xm,Xm,sp_ir, lambda_ir, mask, tol, optionsEP);
    X_EPIR(:,:,i) = EPIR.Xsol;
  end
  Parsol{3} = X_EPIR;

  % % ---------- SCP ADMM ---------- 
  for i=1:3
    Xm = XM(:,:,i);
    Scp_tau = 10;
    optionsScp.max_iter = 1e3 ;
    %     lambda = norm(Xm,"fro")*1;
    SCP = MC_SCpADMM(Xm,Xm,sp_scp, lambda_scp, mask, 1e-4, optionsScp);
    X_SCP(:,:,i) = SCP.Xsol;
  end
  Parsol{4} = X_SCP;

  % % ---------- FGSRp 0.5 ---------- 
  optionsFGSR.d = ceil(1.5*rt);
  optionsFGSR.alphda = 1e-1;
  optionsFGSR.max_iter = 1e3;
  optionsFGSR.tol = tol;
  %   lambda_fgsr = 1;
  %   sp_fgsr = 0.5;
  for i = 1:3
    Xm = XM(:,:,i);
    %     Xr = MC_FGSR_PALM(Xm,mask,lambda_fgsr,optionsFGSR); % for p=0.5 only
    %     [Xr,~,~] = RPCA_FGSR_ADMM(Xm,5e-2,optionsFGSR); X_FGSR(:,:,i) = Xr;
    FGSR = MC_FGSRp_PALM(Xm,mask,sp_fgsr,lambda_fgsr,optionsFGSR);
    %     FGSR = MC_FGSR_PALM(Xm,mask,lambda_fgsr,optionsFGSR); X_FGSR(:,:,i) = FGSR.Xsol;
    %     Xr = MC_FGSR_PALM(Xm,mask,sp_fgsr,lambda_fgsr,optionsFGSR);
    %     X_FGSR(:,:,i) = Xr.Xsol;
    X_FGSR(:,:,i) = FGSR.Xsol;
  end
  Parsol{5} = X_FGSR;
  img_show.sol = Parsol;
  Tab_img{iter_rt} = img_show;
end
%% save cache to plot and make table
save(strcat("..\exp_cache\Tab_", name_pic, datestr(now,'_yymmdd_hhMM'), ".mat"), "Tab_img",'-mat')
% save("..\exp_cache\ImgRe_RandMask_Table_BestLambda_para.mat","Tab_img",'-mat')
% save("..\exp_cache\ImgRe_best_lena.mat","Tab_img",'-mat')
% save("..\exp_cache\ImgRe_best_R1.mat","Tab_img",'-mat')
%% imshow show
img_show = Tab_img{4};
set(gca,'LooseInset',[0,0, 0.003,0.005] )
figure(1)
imshow(img_show.ori_img)

figure(2)
imshow(img_show.low_img)

figure(3)
imshow(img_show.mask_img)

for i = 1:5
  figure(i+3)
  imshow(img_show.sol{i})
end


