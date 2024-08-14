% Written by Ye Wang 03/2022 E-mail: w773664703@gmail.com
% - M is the observation matrix
% - lambda - regularization parameter, default = 1/sqrt(max(N,M))
% - beta - the augmented lagrangian parameter, default = 10*lambda
% - tol - reconstruction error tolerance, default = 1e-6
% - max_iter - maxibetam number of iterations, default = 1000
function Par = ProxIRNN(X0,M,sp, lambda, mask, tol, options)
%% init
if isfield(options,'max_iter')==0,max_iter = 2e3;
else,max_iter = options.max_iter ;
end

if isfield(options,'eps')==0,epsre = 1e-3;
else,epsre = options.eps ;
end

if isfield(options,'beta')==0,beta = 1.1;
else,beta = options.beta ;
end

if isfield(options,'KLopt')==0,KLopt = 1e-5;
else,KLopt = options.KLopt ;
end

if isfield(options,'Rel')==0
  disp("Calculation of correlation distance...");
else
  ReX = options.Rel;
  spRelErr = [];
end

if isfield(options,'zero')==0,zero = 1e-16;
else,zero = options.zero;   % thresholding
end
%% save objective value for ploting
spRelDist = []; spf = [];
sprank = [];
Ssim = []; Rsim = [];
[nr,nc] = size(M) ; rc = min(nr,nc) ;
weps = ones(rc,1)*epsre;
Gradf = @(X)(mask.*(X-M)) ;
Objf = @(x)(norm(mask.*(x-M),'fro')^2/2 + lambda*norm(svds(x,rank(x)),sp)^(sp));
ALF = @(x,y)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svd(x)+y,sp)^(sp));
iter = 0; % Par.f = Objf(X0);
sigma = svd(X0); % ch1
iter = 0;

tic;
while iter <= max_iter
  iter = iter + 1;
  spf(iter) = Objf(X0);
  [U,S,V] = svd(X0 - Gradf(X0)/beta,'econ') ;
  NewS = diag(S) - lambda*sp*(sigma+weps).^(sp-1)/beta ;
  idx = NewS>zero; Rk = sum(idx);
  X1 = U*spdiags(NewS.*idx,0,rc,rc)*V';
  sigma = sort(NewS.*idx,'descend'); % update the sigma

  % save for plot
  sprank(iter) = rank(X1);
  Stime(iter) = toc; % recored the computing time

  % optional information for the iteration
  %     Rsim(iter) = (Objf(X1)-Objf(X0))/(norm(X1-X0,'fro')^2);
  %     Ssim(iter) = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
  %       lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(X1-X0,'fro');
  %     GMinf(iter) = norm(Gradf(X1),inf);
  %% ---------------------- Optimal Condition ----------------------
  opt_sprintf = 'PIRNN satisfies the optimality condition: ';
  RelDist = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
    lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(X1,'fro');
  %       lambda*sp*spdiags((weps(idx)+NewS(idx)).^(sp-1),0,Rk,Rk),'fro')/norm(X1,'fro');
  spRelDist(iter) = RelDist;
  KLdist = norm(X1-X0,inf);
  if exist('ReX','var')
    Rtol = norm(X1-ReX,'fro')/norm(ReX,'fro');
    spRelErr(iter) = Rtol;
    if Rtol < tol
      fprintf( [opt_sprintf,'Relative error\n',...
        'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n'], ...
        iter, Rtol, rank(X1),Objf(X1));
      break;
    end
  end

  %     subpartial = lambda*sp*[(weps(idx)+NewS(idx)).^(sp-1);zeros(min(nr,nc)-Rk,1)];
  %     RelDist = norm(U'*Gradf(X1)*V + spdiags(subpartial,0,Rk,Rk), 'fro')/norm(M,'fro');

  if RelDist<tol
    fprintf( [opt_sprintf, 'Relative Distance\n', ...
      'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n'], ...
      iter, RelDist, rank(X1),Objf(X1));
    break
  end


  if KLdist < KLopt
    fprintf( [opt_sprintf, 'KL optimality condition\n',...
      'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n'], ...
      iter, KLdist, rank(X1),Objf(X1))
    break
  end

  if iter == max_iter
    disp("");
    fprintf( ['PIRNN reachs the MAX_ITERATION \n',...
      'iter:%04d\t rank(X):%d\t Obj(F):%d\n'], ...
      iter, rank(X1),Objf(X1) );
    break
  end

  X0 = X1;  % update the iteration
end % end while

%% return the best-lambda, time, iterations, rank, objective,and solution
if exist('ReX','var')
  Par.RelErr = spRelErr(1:iter);
end
Par.time = Stime;
Par.f = spf(1:iter);
Par.rank = sprank(1:iter);

Par.RelDist = spRelDist(1:iter);

Par.Obj = Objf(X1);
Par.Xsol = X1;
Par.iterTol = iter ;

%   Par.S = Ssim; Par.R = Rsim;
%   Par.GMinf = GMinf;
%   Par.KLdist = KLdist;
end