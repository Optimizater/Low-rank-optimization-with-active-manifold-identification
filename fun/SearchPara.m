% Written by Ye Wang 03/2022(E-mail: w773664703@gmail.com)
% INPUT ::
%   - rel: the ture image or matrix
%   - X0: start point
%   - Xm: the observation matrix
%   - Method: the different method for low rank solver
%   - lambdalist: the search list for lambda
%   - splist: the search list for sp
%   - mask: the mask matrix
%   - tol: reconstruction error tolerance
%   - options: different options for different method
%     - max_iter - maxibetam number of iterations, default = 500
% OUTPUT ::
%   - LAMBDA: the best lambda 
%   - SP: the best sp 
function [LAMBDA, SP] = SearchPara(rel, X0, Xm, Method, lambdalist, splist, mask, tol, options)


if length(lambdalist) < 2 && length(splist) < 2
  LAMBDA.opt_index = 1;
  LAMBDA.opt_para = lambdalist;
  LAMBDA.opt_loc = [-1/0, 1/0];

  SP.opt_index = 1;
  SP.opt_para = splist;
  SP.opt_loc = [-1/0, 1/0];
  return
end

%% search lambda
sp = splist(ceil(length(splist)/2));       
Lambda_PSNR = zeros(length(lambdalist),1);
if length(lambdalist) > 1
%   for idx_lambda = 1 : length(lambdalist)
  parfor idx_lambda = 1 : length(lambdalist)
    switch Method
      case 'SCP'
        SearchR = MC_SCpADMM(X0, Xm, sp, lambdalist(idx_lambda), mask, tol, options);
      case 'PIR'
        SearchR = ProxIRNN(X0, Xm, sp, lambdalist(idx_lambda), mask, tol, options);
      case 'AIR'
        SearchR = AIRNN_2021(X0, Xm, sp, lambdalist(idx_lambda), mask, tol, options);
      case 'EIR'
        SearchR = EIRNRI(X0, Xm, sp, lambdalist(idx_lambda), mask, tol, options);
      case 'FGSR'
        SearchR = MC_FGSR_PALM(Xm, mask, sp, lambdalist(idx_lambda), options);
      otherwise
        error('SearchPara does not has such method!')
    end
    Lambda_PSNR(idx_lambda) = psnr(rel, SearchR.Xsol);
  end

  [~, optLambdaIdx] = max(Lambda_PSNR(:,1));
  opt_lambda = lambdalist(optLambdaIdx);
  LAMBDA.opt_index = optLambdaIdx;
  LAMBDA.opt_para = opt_lambda;
  LAMBDA.par_list = lambdalist;
  LAMBDA.PSNR = Lambda_PSNR;
  if optLambdaIdx == 1
    if Lambda_PSNR(1,1) == Lambda_PSNR(2,1)
      LAMBDA.opt_loc = [lambdalist(1), lambdalist(2)];
    else
      LAMBDA.opt_loc = [-1/0, lambdalist(1)];
    end
  elseif optLambdaIdx == length(lambdalist)
    if Lambda_PSNR(end-1,1) == Lambda_PSNR(end,1)
      LAMBDA.opt_loc = [lambdalist(end-1), lambdalist(end)];
    else
      LAMBDA.opt_loc = [lambdalist(end), 1/0];
    end
  else
    LAMBDA.opt_loc = [lambdalist(optLambdaIdx-1), lambdalist(optLambdaIdx+1)];
  end
else
  %% length(lambdalist) == 1
  LAMBDA.opt_index = 1;
  LAMBDA.opt_para = lambdalist;
  LAMBDA.par_list = lambdalist;
  LAMBDA.PSNR = 0;
  LAMBDA.opt_loc = [-1/0, 1/0];
end

%% search sp
Lambda = LAMBDA.opt_para;
sp_PSNR = zeros(length(splist), 1);
if length(splist) > 1
%   for idx_sp= 1 : length(splist)
  parfor idx_sp= 1 : length(splist)
    switch Method
      case 'SCP'
        SearchR = MC_SCpADMM(X0, Xm, splist(idx_sp), Lambda, mask, tol, options);
      case 'IR'
        SearchR = EIRNRI(X0, Xm, splist(idx_sp), Lambda, mask, tol, options);
      case 'FGSR'
        SearchR = MC_FGSRp_PALM(Xm, mask, splist(idx_sp), Lambda, options);
      otherwise
        error('SearchPara does not has such method!')
    end
    sp_PSNR(idx_sp) = psnr(rel, SearchR.Xsol);
  end

  [~, optspIdx] = max(sp_PSNR(:,1));
  opt_sp = splist(optspIdx);
  SP.opt_index = optspIdx;
  SP.opt_para = opt_sp;
  SP.par_list = splist;
  SP.PSNR = sp_PSNR;
  if optspIdx == 1
    if sp_PSNR(1,1) == sp_PSNR(2,1)
      SP.opt_loc = [splist(1), splist(2)];
    else
      SP.opt_loc = [-1/0, splist(1)];
    end
  elseif optspIdx == length(splist)
    if sp_PSNR(end-1,1) == sp_PSNR(end,1)
      SP.opt_loc = [splist(end-1), splist(end)];
    else
      SP.opt_loc = [splist(end), 1/0];
    end
  else
    SP.opt_loc = [splist(optspIdx-1), splist(optspIdx+1)];
  end
else
  SP.opt_index = 1;
  SP.opt_para = splist;
  SP.par_list = splist;
  SP.PSNR = 0;
  SP.opt_loc = [-1/0, 1/0];
end
end
