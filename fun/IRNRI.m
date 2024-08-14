% Written by Ye Wang 03/2022(E-mail: w773664703@gmail.com)
% INPUT ::
% - X0: start point
% - M: the observation matrix
% - sp: the Schatten-p norm
% - lambda: regularization parameter
% - mask: mask matrix
% - tol: reconstruction error tolerance
% - options
%   - max_iter - maxibetam number of iterations, default = 2000
% OUTPUT ::
% - Par
%   - Xsol: restored matrix
function Par = IRNRI(X0,M,sp, lambda, mask, tol, options)
%% init
if isfield(options,'max_iter')==0,max_iter = 2e3;
else,max_iter = options.max_iter;
end

if isfield(options,'eps')==0,epsre = 1;
else,epsre = options.eps;
end

if isfield(options,'beta')==0,beta = 1.1;
else,beta = options.beta;
end

if isfield(options,'KLopt')==0,KLopt = 1e-5;
else,KLopt = options.KLopt;
end

if isfield(options,'mu')==0,mu = 0.5;
else,mu = options.mu;
end

if isfield(options,'Rel')==0
  disp("Calculation of correlation distance...");
else
  ReX = options.Rel;
  spRelErr = [];
end

if isfield(options,'zero')==0,zero = 1e-16;
else,zero = options.zero;   % thresholding for the minimum singular value
end

if isfield(options,'teps')==0,teps = 1e-16;
else,teps = options.teps;   % restrict the weighted epsilon for AdaIRNN
end

% --- objective function ---
if isfield(options,'objf')==0
  Objf = @(x)(norm(mask.*(x-M),'fro')^2/2 + lambda*norm(svds(x,rank(x)),sp)^(sp));
else  
  % demo: 
  % Objf = '@(x)(x-1)'
  Objf = str2func(options.Objf);
end

% --- first order derivative ---
if isfield(options,'Gradf')==0
  Gradf = @(X)(mask.*(X-M));
else
  Gradf = str2func(options.Gradf);
end

ALF = @(x,y)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svd(x)+y,sp)^(sp));
%% save objective value for ploting
spRelDist = []; spf = [];
sprank = [];
Ssim = []; Rsim = [] ;
[nr,nc] = size(M); rc = min(nr,nc);
weps = ones(rc,1)*epsre;

iter = 0; %Par.f = Objf(X0);
Rk0 = rank(X0);
sigma = svd(X0); % ch1
iter = 0;

tic;
while iter < max_iter
  iter = iter + 1;
  spf(iter) = Objf(X0);
  [U,S,V] = svd(X0 - Gradf(X0)/beta,'econ');

  NewS = diag(S) - lambda*sp*(sigma+weps).^(sp-1)/beta;
  NewS(isinf(NewS)) = 0;
  idx = NewS>zero; Rk1 = sum(idx);

  weps = update_eps(weps,Rk0,Rk1,rc,NewS(Rk1),mu);
  X1 = U*spdiags(NewS.*idx,0,rc,rc)*V';

  % restrict the eps
  if isfield(options,"teps")
    weps = (weps<teps) .* teps + (weps>=teps) .* weps;
  end

  sigma = sort(NewS.*idx,'descend'); % update the sigma
  % save for plot
  sprank(iter) = rank(X1);
  Stime(iter) = toc; % recored the computing time

  % optional information for the iteration
  %     Rsim(iter) = (Objf(X1)-Objf(X0))/(norm(X1-X0,'fro')^2);
  %     Ssim(iter) = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
  %       lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(X1-X0,'fro');
  %     GMinf(iter) = norm(Gradf(X1),inf);

  % The Initialization Information
  %     if iter==1
  %       fprintf(1, 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
  %               iter, RelDist, rank(X1),Objf(X1) );
  %     end
  %% ---------------------- Optimal Condition ----------------------
  opt_sprintf = 'IRNRI satisfies the optimality condition: ';
  RelDist = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
    lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk1,Rk1),'fro')/norm(X1,'fro');
  %       lambda*sp*spdiags((weps(idx)+NewS(idx)).^(sp-1),0,Rk,Rk),'fro')/norm(X1,'fro');
  spRelDist(iter) = RelDist;
  KLdist = norm(X1-X0,inf);
  if exist('ReX','var')
    Rtol = norm(X1-ReX,'fro')/norm(ReX,'fro');
    spRelErr(iter) = Rtol;
    if Rtol <= tol
      fprintf( [opt_sprintf, 'Relative error\n',...
        'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n'], ...
        iter, Rtol, rank(X1),Objf(X1));
      break;
    end
  end

  if RelDist < tol
    fprintf( [opt_sprintf, 'Relative Distance\n',...
      'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n'], ...
      iter, RelDist, rank(X1),Objf(X1));
    break
  end

  %     KLdist = norm(X1-X0,"fro")+(1-mu)*norm(weps(1:Rk),1)/mu;
  if KLdist < KLopt
    fprintf( [opt_sprintf, 'KL optimality condition\n',...
      'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n'], ...
      iter, KLdist, rank(X1),Objf(X1))
    break
  end

  %     if norm(mask.*(X1-M),inf)<tol
  %       disp("Iteration terminates");
  %       fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
  %           iter, RelDist, rank(X1),Objf(X1));
  %     end

  if iter == max_iter
    fprintf( ['IRNRI reachs the MAX_ITERATION \n',...
      'iter:%04d\t rank(X):%d\t Obj(F):%d\n'], ...
      iter, rank(X1),Objf(X1) );
    break
  end

  %     if mod(iter,1000)==0
  %        fprintf( 'iter:%04d\t rank(X):%d\t Obj(F):%d\n', ...
  %           iter, rank(X1),Objf(X1) );
  %     end

  X0 = X1 ; % update the iteration
  Rk0 = Rk1;
end % end while

%% return the best-lambda, time, iterations, rank, objective,and solution
if exist('ReX','var')
  Par.RelErr = spRelErr(1:iter);
end

Par.time = Stime;
Par.f = spf;
Par.rank = sprank(1:iter);

Par.RelDist = spRelDist(1:iter);

Par.Obj = Objf(X1);
Par.Xsol = X1;
Par.iterTol = iter;

Par.weps = weps(1:Rk1);
%   Par.S = Ssim; Par.R = Rsim;
%   Par.GMinf = GMinf;
Par.KLdist = KLdist;
end