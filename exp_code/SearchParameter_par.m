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


name_pic = 're1'; %'sist_gym';
name_suffix = '.jpg';
path_img = strcat(cwd,'\img_image\', name_pic, name_suffix );    % % select the pic
img_ori = double(imread(path_img))/255 ;
it_mask = 2;  % % mask for different methods-> 1: random, 2:column, 3:square
missrate = 0.5; % sampleRate = 1 - missRate     % random mask
if_img_show = 0;
%% To search a better parameters.
scan.p = 0;
scan.lambda = 0;
scan.deep = 2; % the times of recursively searching
if isempty(gcp('nocreate'))
  %   parpool();
end

img_size = size(img_ori);
% noise = 1e0*randn(img_size(1:2));

%% ---------- get mask ----------
if exist('it_mask','var')==0 || it_mask == 1
  % random mask
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

%% ---------- strictly low rank ----------
RT = [15,20,25,30,35,40];
sol_method = ["PIR", "AIR", "EIR", "SCP", "FGSR"];
% ------------ parameters for different method ------------
tol = 1e-5;
max_iter = 3e3;
max_iter_scp = 1e3;

Klopt = 1e-5;
options.max_iter = max_iter;


optionsSCP.max_iter = max_iter_scp; % 1e3

optionsEP = options;
optionsEP.mu = 0.7;
optionsEP.eps = 1;

optionsFGSR.regul_B = "L21";

optMethod{1} = options;
optMethod{2} = options;
optMethod{3} = optionsEP;
optMethod{4} = optionsSCP;
optMethod{5} = optionsFGSR;

% for iter_rt = 1:1
search_path_deep = {}; % record the search path
search_path = zeros(length(sol_method), 1);

%% --------------- RECOVERY search paras ------------------------
for iter_rt = 1:1%length(RT)
  rt = RT(iter_rt);
  for i = 1:3
    [U,S,V] = svd(img_ori(:,:,i));
    Xt(:,:,i) = U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end

  % start point + extrapolation coefficient + parameters
  XM = mask.*Xt;
  %   X_INIT_0 = zeros(img_size(1:2));
  %   X_INIT_RAND1 = (1+randn(img_size(1),rt))*(randn(rt,img_size(2)));  % generate initial point randomily

  %   options.KLopt = 1e-5;
  %% ------------------% search lambda -----------------------------

  if exist('scan','var') && isfield(scan,'lambda') && scan.lambda == 1
    % search with fixed sp
    sp = 0.5;
    scan.linspace = [0, 1]; % the first time has the lambda_max, and then begin with "lambda_best * eta"
    scan.skip = [7, 9]; % linspace of search scope
    scan.linearSearch = [0.5^4, 0.5]; % scan.skip = 2 * log{linearSearch(i)}{linearSearch(i+1)} - 1;
    % % 2^8 ~ 2^-20=> k=0,...,7
    % %
    scanM_method = 2^8 * ones(length(sol_method), 1);  % search from the same init points

    for iscan = 1 : scan.deep
      for it_method = 2 : length(sol_method) % do not need parfor
        Lambda_Slist = scanM_method(it_method) .* scan.linearSearch(iscan).^(scan.linspace(iscan) : 1 : scan.skip(iscan));
        [LAMBDA_out, ~] = SearchPara(img_ori(:,:,1), XM(:,:,1), XM(:,:,1), sol_method(it_method), Lambda_Slist, sp, mask, tol, optMethod{it_method});
        lambda_best = LAMBDA_out.opt_para;
        scanM_method(it_method) = lambda_best ./ scan.linearSearch(iscan);
        search_path(it_method) = lambda_best;
      end
      search_path_deep{iscan}(:, iter_rt) = search_path;
    end
  else
    lambda_pir = 0.5;
    lambda_air = lambda_pir;
    lambda_eir = lambda_pir;
    lambda_scp = lambda_pir;
    lambda_fgsr = lambda_pir;
  end
  %% ------------------ search p-----------------------------
  % search with fixed lambda
  if exist('scan','var') && isfield(scan,'p') && scan.p == 1
    optionsEP.alpha = 0.7;

    op = 0.01 : 0.03 : 1; % % [0,1] = 0.01, 0.04, ... , 0.01 + 0.03*pidx

    [~, SP_SCP] = SearchPara(img_ori(:,:,1), XM(:,:,1), XM(:,:,1), 'SCP', lambda_scp, op, mask, tol, optionsSCP);
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
end % end for different rank
%% save the best parameters or reload the best parameters
% save("..\exp_cache\BestSearchLambda_RandMask5.mat","search_path_deep",'-mat')
if ~isempty(search_path_deep)
  save(strcat("..\exp_cache\BestSearchLambda_", name_pic, '_mask', num2str(it_mask), datestr(now,'_yymmddhhMM'), ".mat"), "search_path_deep",'-mat');
else
  para = dir('..\exp_cache');
  cand_name = strcat("BestSearchLambda_", name_pic, '_mask', num2str(it_mask));
  cand_name_len = strlength(cand_name);
  cand_exist = 0;
  for it_file = 1 : length(para)
    if strncmpi(cand_name, para(it_file).name, cand_name_len)
      disp(it_file)
      cand_exist = 1;
      break
    end
  end
  if cand_exist
    load(strcat('../exp_cache/', para(it_file).name));
  else
    error('It can not find the best parameters, search the best parameters first!')
  end
end
%% ------------------------- RECOVERY image -------------------------
% --------------------------different method show-------------------------------
% ------------------------------------------------------------------------------
for iter_rt = 1:length(RT)
  lambda_pir = search_path_deep{length(scan.deep)}(1, iter_rt);
  lambda_air = search_path_deep{length(scan.deep)}(2, iter_rt);
  lambda_eir = search_path_deep{length(scan.deep)}(3, iter_rt);
  lambda_scp = search_path_deep{length(scan.deep)}(4, iter_rt);
  lambda_fgsr = search_path_deep{length(scan.deep)}(5, iter_rt);
  rt = RT(iter_rt);
  parfor i = 1:3
    [U,S,V] = svd(img_ori(:,:,i));
    Xt(:,:,i) = U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end
  XM = mask.*Xt;
  %% ---------- parameter for different model ----------
  % PIRNN
  optionsP = options;

  % AIRNN
  optionsA = optionsP;
  optionsA.mu = 0.7;
  %   optionsA.eps = 0.5;

  % IRNRI
  optionsEP = optionsA;
  optionsEP.alpha = 0.7;
  %% ----------- recovery image by different method -----------
  img_show.ori_img = img_ori;
  img_show.low_img = Xt;
  img_show.mask_img = XM;
  Parsol = {};
  % ---------- PIRNN ----------
  parfor i =1:3
    Xm = XM(:,:,i);
    PIR = ProxIRNN(Xm,Xm,sp_ir, lambda_pir, mask, tol, optionsP);
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
  parfor i=1:3
    Xm = XM(:,:,i);
    %     options.Rel = img_ori(:,:,i);
    AIR = AIRNN_2021(Xm,Xm,sp_ir, lambda_air, mask, tol, options);
    X_AIR(:,:,i) = AIR.Xsol;
  end
  Parsol{2} = X_AIR;

  % % ---------- IRNRI ----------
  parfor i=1:3
    Xm = XM(:,:,i);
    EPIR = EIRNRI(Xm,Xm,sp_ir, lambda_eir, mask, tol, optionsEP);
    X_EPIR(:,:,i) = EPIR.Xsol;
  end
  Parsol{3} = X_EPIR;

  % % ---------- SCP ADMM ----------
  parfor i=1:3
    Xm = XM(:,:,i);
    %     Scp_tau = 10; dummy parameters
    %     lambda = norm(Xm,"fro")*1;
    SCP = MC_SCpADMM(Xm,Xm,sp_scp, lambda_scp, mask, 1e-4, optionsSCP);
    X_SCP(:,:,i) = SCP.Xsol;
  end
  Parsol{4} = X_SCP;

  % % ---------- FGSRp 0.5 ----------
  optionsFGSR.d = ceil(1.5*rt);
  optionsFGSR.alphda = 1e-1;
  optionsFGSR.max_iter = max_iter;
  optionsFGSR.tol = tol;
  %   lambda_fgsr = 1;
  %   sp_fgsr = 0.5;
  parfor i = 1:3
    Xm = XM(:,:,i);
    %     Xr = MC_FGSR_PALM(Xm,mask,lambda_fgsr,optionsFGSR); % for p=0.5 only
    %     [Xr,~,~] = RPCA_FGSR_ADMM(Xm,5e-2,optionsFGSR); X_FGSR(:,:,i) = Xr;
    FGSR = MC_FGSR_PALM(Xm,mask,sp_fgsr,lambda_fgsr,optionsFGSR);
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
save(strcat("..\exp_cache\Tab_", name_pic, '_mask', num2str(it_mask), datestr(now,'_yymmddhhMM'), ".mat"), "Tab_img",'-mat')
% save("..\exp_cache\ImgRe_RandMask_Table_BestLambda_para.mat","Tab_img",'-mat')
% save("..\exp_cache\ImgRe_best_lena.mat","Tab_img",'-mat')
% save("..\exp_cache\ImgRe_best_R1.mat","Tab_img",'-mat')
%% imshow show
if if_img_show
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

  figure()
  for i = 1:5
    subplot(2,3,i)
    imshow(img_show.sol{i})
  end
end
datetime('now')