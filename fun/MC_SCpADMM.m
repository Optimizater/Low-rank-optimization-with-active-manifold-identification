% Modified by Ye Wang 03/2022 E-mail: wangye@shanghaitech.edu.cn
% -X0 is the sampling matrix
% -M is the mask matrix
% ----
function Par = MC_SCpADMM(X0, M, sp, lambda, mask, tol, opt)

if isfield(opt,'max_iter')==0,max_iter = 100;
else,max_iter = opt.max_iter ;
end

% Scp norm thresholder
% tua=0 means the Schatten-p norm
if isfield(opt,'tau')==0, opt.tau = 30 ;
else,opt.tau = opt.tau;
end

spf = []; sprank = [];


tic ;
opt.lambda = lambda; % regularization parameter
opt.p = sp; % Scp norm
opt.omega = mask; % observed set
opt.D_omega = M; % sampling set

Objf = @(x)(norm(mask.*(x-M),'fro')^2/2 + lambda*norm(svds(x,rank(x)),sp)^(sp));

opt.omega = mask;
opt.D_omega = M;

Y_omega = opt.D_omega;
E_omega = opt.D_omega;
W = opt.D_omega;
Z = Y_omega;

rou = 1.5;
mv = 1.5;

for iter = 1:max_iter
  X = opt_X(E_omega,Y_omega,W,Z,mv,opt);
  E_omega = opt_E(X,Y_omega,mv,opt);
  W = opt_W(X,Z,mv,opt);
  Y_omega = Y_omega + mv*(E_omega - X .* opt.omega + opt.D_omega);
  Z = Z + mv*(X - W);
  mv = rou*mv;
  Stime(iter) = toc;
  spf(iter) = Objf(X);
  sprank(iter) = rank(X);
end
X1 = X;

if iter==max_iter
  disp("SCP Reach the MAX_ITERATION");
  fprintf( 'iter:%04d\t  rank(X):%d\t Obj(F):%d\n', ...
    iter, rank(X1),Objf(X1) );
end

if norm(X1-X0,"fro")<tol || norm(mask.*(X1-M))<tol
  disp("SCP Satisfying the optimal condition");
  fprintf( 'iter:%04d\t  rank(X):%d\t Obj(F):%d\n', ...
    iter, rank(X1),Objf(X1) );
end

X0 = X1;

estime = toc;

Par.time = Stime(1:iter);
Par.f = spf(1:iter) ;
Par.rank = sprank(1:iter);
Par.iterTol = iter ;

Par.Xsol = X1;
end