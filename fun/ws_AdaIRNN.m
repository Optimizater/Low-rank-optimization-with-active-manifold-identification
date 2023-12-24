function Par = ws_AdaIRNN(X0,M,sp, lambda, mask, tol, options)
  % - M is the observation matrix
  % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
  % - beta - the augmented lagrangian parameter, default = 10*lambda
  % - tol - reconstruction error tolerance, default = 1e-6
  % - max_iter - maxibetam number of iterations, default = 1000
 
  if isfield(options,'max_iter')==0,max_iter = 2e3;
  else,max_iter = options.max_iter ;
  end
  
  if isfield(options,'eps')==0,epsre = 1;
  else,epsre = options.eps ;
  end
  
  if isfield(options,'beta')==0,beta = 1.1;
  else,beta = options.beta ;
  end
  
  if isfield(options,'KLopt')==0,KLopt = 1e-5*min(size(M));
  else,KLopt = options.KLopt ;
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
  
  if isfield(options,'zero')==0,zero = 1e-2;
  else,zero = options.zero;   % thresholding
  end

  spRelDist = []; spf = [];
  sprank = [];
  Ssim = []; Rsim = [] ; 
  [nr,nc] = size(M); rc = min(nr,nc); 
  weps = ones(rc,1)*epsre; 
  
  Gradf = @(X)(mask.*(X-M)); 
%   Objf = @(x)(norm(mask.*(x-M),'fro')^2/2 + lambda*norm(svds(x,rank(x)),sp)^(sp)); 
%   ALF = @(x,y)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svd(x)+y,sp)^(sp));

  iter = 0; %Par.f = Objf(X0);
  sigma = svd(X0); % ch1

  tic ;
  lambda_Target = lambda;
  lambda = 1e4 * lambda;
  lambda_rho = 0.8;
  iter = 0;
  sweepTimes = floor(ceil(log(1e-4)/(log(lambda_rho))));
  insweep = max(ceil(max_iter/sweepTimes),200); 

  while lambda > lambda_Target && iter < max_iter
    Objf = @(x)(norm(mask.*(x-M),'fro')^2/2 + lambda*norm(svds(x,rank(x)),sp)^(sp));
    ALF = @(x,y)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svd(x)+y,sp)^(sp));
%     weps = ones(rc,1)*epsre;
    for ins = 1:1:insweep
      iter = iter + 1;
      spf(iter) = Objf(X0);
      [U,S,V] = svd(X0 - Gradf(X0)/beta,'econ');
% restart the eps
%     if ~isempty(find(and(diag(S)>zero,(sigma+weps)<zero),1)) && (iter<=1e2)
%       weps(and(diag(S)>zero,(sigma+weps)<zero)) = epsre;
%     end 
      
      NewS = diag(S) - lambda*sp*(sigma+weps).^(sp-1)/beta;
      NewS(isinf(NewS)) = 0; 
      idx = NewS>zero; Rk = sum(idx);  
      X1 = U*spdiags(NewS.*idx,0,rc,rc)*V'; 
      weps(weps(1:Rk)>zero) = weps(weps(1:Rk)>zero)*mu ;
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

      if exist('ReX','var')
        Rtol = norm(X1-ReX,'fro')/norm(ReX,'fro');
        Rate(iter) = norm((X1-ReX),'fro')/norm((ReX),'fro');
        spRelErr(iter) = Rtol;
        if Rtol<=tol
          break;  
        end
      end

      RelDist = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
        lambda*sp*spdiags((weps(idx)+NewS(idx)).^(sp-1),0,Rk,Rk),'fro')/norm(M,'fro'); 
      spRelDist(iter) = RelDist; 

      if RelDist < tol
        break
      end

      KLdist = norm(X1-X0,"fro");
  %     KLdist = norm(X1-X0,"fro")+(1-mu)*norm(weps(1:Rk),1)/mu;
      if KLdist<KLopt
        break
      end

      X0 = X1 ; % update the iteration
    end %^ end for sweep
    lambda = lambda*lambda_rho; 
  end % end while -- lambda & iter    
    %% ---------------------- Optimal Condition ----------------------
    if exist('ReX','var')
%       Rtol = norm(X1-ReX,'fro')/norm(ReX,'fro');
%       Rate(iter) = norm(mask.*(X1-ReX),'fro')/norm(mask.*(ReX),'fro');
%       spRelErr(iter) = Rtol;
      if Rtol <= tol
        disp('AdaIRNN: Satisfying the optimality condition:Relative error'); 
        fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelErr, rank(X1),Objf(X1));
%         break;  
      end
    end
    
%     RelDist = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
%       lambda*sp*spdiags((weps(idx)+NewS(idx)).^(sp-1),0,Rk,Rk),'fro')/norm(M,'fro'); 
%     spRelDist(iter) = RelDist; 
    if RelDist<tol
      disp('AdaIRNN: Satisfying the optimality condition:Relative Distance'); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X1),Objf(X1));
%       break
    end
    
%     KLdist = norm(X1-X0,"fro");
%     KLdist = norm(X1-X0,"fro")+(1-mu)*norm(weps(1:Rk),1)/mu;

    if KLdist < KLopt
      disp("AdaIRNN: Satisfying  the KL optimality condition"); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, KLdist, rank(X1),Objf(X1))
%       break
    end

%     if norm(mask.*(X1-M),inf)<tol
%       disp("Iteration terminates");
%       fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
%           iter, RelDist, rank(X1),Objf(X1));
%     end
       
    if iter==max_iter
      disp("AdaIRNN: Reach the MAX_ITERATION");
      fprintf( 'iter:%04d\t rank(X):%d\t Obj(F):%d\n', ...
        iter, rank(X1),Objf(X1) );
%       break
    end
    
%     if mod(iter,1000)==0
%        fprintf( 'iter:%04d\t rank(X):%d\t Obj(F):%d\n', ...
%           iter, rank(X1),Objf(X1) );
%     end
% update the iteration 
%     X0 = X1 ;   
%   end % end while 
  estime = toc;

%% return the best-lambda, time, iterations, rank, objective,and solution
  if exist('ReX','var')
    Par.RelErr = spRelErr(1:iter); 
    Par.Rate = Rate;
  end

  Par.time = Stime;
  Par.f = spf;
  Par.rank = sprank(1:iter);
  
  Par.RelDist = spRelDist(1:iter); 
  
  Par.Obj = Objf(X1);
  Par.Xsol = X1; 
  Par.iterTol = iter;
  
  Par.weps = weps(1:Rk);
%   Par.S = Ssim; Par.R = Rsim;
%   Par.GMinf = GMinf;
  Par.KLdist = KLdist;
end