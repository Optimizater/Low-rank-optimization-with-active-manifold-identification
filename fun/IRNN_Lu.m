% Modified by Ye Wang 2022/05(E-mail: wangye@shanghaitech.edu.cn)
% INPUT ::
% - X0: is the Initial point, should be a vector with m*n 
% - fun: the regularization function
% - y: the observation matrix with linear operator
% - M: 
% - m: 
% - n: 
% - options
% OUTPUT ::
% - Par
% Written by Canyi Lu (canyilu@gmail.com)
% References: 
% Canyi Lu, Jinhui Tang, Shuicheng Yan and Zhouchen Lin,
% Nonconvex Nonsmooth Low Rank Minimization via Iteratively Reweighted Nuclear Norm,
% IEEE Transactions on Image Processing, vol. 25, pp. 829-839, 2016
% % Canyi Lu, Jinhui Tang, Shuicheng Yan and Zhouchen Lin,
% Generalized Nonconvex Nonsmooth Low-Rank Minimization,
% International Conference on Computer Vision and Pattern Recognition (CVPR), 2014
function Par = IRNN_Lu(X0,fun,y,M,m,n,options)
%% init
  if isfield(options,"max_iter"), max_iter = options.max_iter;
  else, max_iter = 5e3;
  end

  if isfield(options,"gamma"), gamma = options.gamma ; % Schatten norm
  else, gamma = 1;
  end
  
  if isfield(options,"lambda_Init"), lambda = options.lambda_Init;
  else, lambda = max(abs(M(y,2)));
  end
  
  if isfield(options,"lambda_rho"), lambda_rho = options.lambda_rho;
  else, lambda_rho = 0.9;
  end
  
  if isfield(options,"lambda_Target"), lambda_Target = options.lambda_Target;
  else, lambda_Target = max(abs(M(y,2)))*1e-5;
  end


  if isfield(options,"tol"), tol = options.tol ;
  else, tol = 1e-5;
  end

  if isfield(options,"mu"), mu = options.mu ;
  else, mu = 1.1;
  end

%% save objective value for ploting
  Objf = @(x,X)(norm(y-M(x,1),2)^2/2 +...
    lambda_Target*norm(svds(X,rank(X)),gamma)^(gamma));
  hfun_sg = str2func([fun '_sg']);
  
%   x = zeros(m*n,1);
% Initial point 
  x = X0(:);
  X = reshape(x,[m,n]);
  %% with warm start  
  insweep = 2e2; % warm  start step
  iter = 0;
  tic;
  while lambda > lambda_Target && iter < max_iter
    ftol = @(x)(norm(y-M(x,1)) + lambda*norm(x,1));
    f_current = ftol(x) ; 
%     ins = 0;
    for ins = 1: 1: insweep
      iter = iter+1; 
%       ins = ins+1;
      % save for plot and table       
      Stime(iter) = toc;
      sprank(iter) = rank(X);
      spf(iter) = Objf(x,X);
      
      f_previous = f_current;
      x = x + (1/mu)*M(y - M(x,1),2);
      [U,S,V] = svd(reshape(x,[m,n]),'econ');
      sigma = diag(S);
      w = hfun_sg(sigma,gamma,lambda);
      sigma = sigma - w/mu;
      svp = length(find(sigma>0));
      sigma = sigma(1:svp);
      X = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
      x = X(:);
      f_current = norm(y-M(x,1)) + lambda*norm(x,1);
      if norm(f_current-f_previous)/norm(f_current + f_previous) < tol
        break;
      end
    end
    
    if norm(y-M(x,1)) < tol
      disp("Satisfying the optimal condition"); break; 
    end
    
    if iter==max_iter
      disp("Reach the MAX_ITERATION");
      fprintf( 'iter:%04d\t rank(X):%d\t Obj(F):%d\n',iter, rank(X),Objf(X) );
      break
    end
    lambda = lambda*lambda_rho; 
  end
%% without warm start
% % % % skip to optimal lambda to solve
%   iter = 0; lambda = lambda_Target; 
%   x0 = x;
%   tic;
%   while iter<max_iter
%     iter = iter + 1;
%     
%     Stime(iter) = toc;
%     sprank(iter) = rank(X);
%     spf(iter) = Objf(x,X);
%     
%     x = x + (1/mu)*M(y - M(x,1),2);
%     [U,S,V] = svd(reshape(x,[m,n]),'econ');
%     sigma = diag(S);
%     w = hfun_sg(sigma,gamma,lambda);
%     sigma = sigma - w/mu;
%     svp = length(find(sigma>0));
%     sigma = sigma(1:svp);
%     X = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
%     x1 = X(:);
%     
% % Termination condition
%     if norm(y-M(x1,1)) < tol || norm(x1-x0,2)<tol
%       disp("IRNN_Lu: Satisfying the optimal condition")
%       fprintf( 'iter:%04d\t err%06f\t  rank(X):%d\t Obj(F):%d\n', ...
%         iter,norm(x1-x0,2), rank(X),Objf(x,X) );
%       break;
%     end
%     
%     if iter==max_iter
%       disp("IRNN_Lu: Reach the MAX_ITERATION");
%       fprintf( 'iter:%04d\t  rank(X):%d\t Obj(F):%d\n', ...
%         iter, rank(X),Objf(x,X) );
%       break
%     end
%     x = x1; x0=x1;% update the ieration
%   end% end while
%% return the best-lambda, time, iterations, rank, objective,and solution
  Par.lambda_best = lambda;
  Par.time = Stime;
  Par.iterTol = iter;
  Par.rank = sprank;
  Par.f = spf;
  Par.Xsol = reshape(x,[m,n]);
end