function X1 = MC_IRSVM(M,X,sp,lambda,mask,tol,options)
    % - X can also contain NaN's for unobserved values 
    % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
    % - tol - reconstruction error tolerance, default = 1e-3
    % - max_iter - maximum number of iterations, default = 1000

    [row, col] = size(M) ;  
    l = min(row,col) ; 
    q = sp/(sp-1) ; 

% ----------- initialization for parameters -----------
    Ff = @(X)(norm(mask.*( X -M),'fro')) ; 
    Fgradf = @(X)(mask.*(X-M)) ; 
    epsr = epsFind(1e-10,1,q,Ff(M),lambda*sp*(l*lambda)^(-1/q)*sqrt(2) ); 
    varell = epsr/(l*lambda) ; 
    Fh = @(t)(min(t.^(q/(q-1)), sp*(t*varell.^(1/q)-varell/q) ) ) ;  
    FF = @(X,t)(norm(mask.*(X-M),'fro')+lambda*sum(Fh(svd(X)))) ; 
    
% ----------- Setting the default value -----------
    if isfield(options,'max_iter')==0,max_iter = 1000;
    else,max_iter = options.max_iter ;
    end

    if isfield(options,'Lipbd')==0,Lipl = 1e-2;Lipr = 1;
      fprintf('The default bound of lipschitz is: %d and %d \n',Lipl,Lipr);
    else,Lipl=options.Lipbd(1); Lipr = options.Lipbd(2);
    end
    
    if isfield(options,'Nre')==0,Nre = 10;
      fprintf('Find the largest in top %d for each sub-iteration \n',Nre);
    else,Nre=options.Nre; 
    end
    
    if isfield(options,'c')==0,c = 1e-4 ; 
    else, c=options.c;
    end

    if isfield(options,'tau')==0,tau = 2 ; 
    else, tau=options.tau;
    end
    
% -----------  iteration  -----------
  RelErr = 1 ; 
  iter = 0 ; 
  Fre = zeros(Nre,1) ; 
  X0 = X ; 
  while RelErr>tol && iter<max_iter
    if iter == 0, Liter = 1 ; 
    else, Liter = max(Lipl,min(Lipr,trace(DX*DGf')/(norm(DGf,'fro')^2))) ; 
    end
    % if iter == 0, Liter = (Lipl+Lipr)/2 ; end
    iter = iter + 1;
% solve the linear-subproblem
    while 1
      sigma0 = svd(X0) ;
      w = min(varell*ones(l,1),sigma0.^(1/(q-1)) ) ; 
      [U,S,V] = svd(X0-Fgradf(X0)/Liter,'econ') ; 
      NewS = diag(S) - sp*w ;
      X1 = U*spdiags(NewS.*(NewS>0),0,l,l)*V' ; 

      FF_new = FF(X1,epsr);  
      if iter <= Nre, Fre(iter) = FF_new ;
      else 
        if FF_new<= max(Fre) - c*norm(X1-X0,'fro')/2
          DX = X1-X0; DGf = Fgradf(X1) - Fgradf(X0); X0 = X1 ; 
          RelErr = norm(X0-M,'fro')/norm(M,'fro') ;
          break
        else, Liter = tau*Liter ;
        end
        Fre(1:9) = Fre(2:10); Fre(10) = FF_new; X0 = X1;
      end
    end
% update the RelErr    
    if (iter == 1) || (mod(iter, 5) == 0) || (RelErr < tol)
      fprintf(1, 'iter: %04d\t err: %f\t rank(X):%d\t Objf:%f\n', ...
                iter, RelErr, rank(X1),Ff(X0)+lambda*norm(svd(X0),sp).^sp );
    end
  end
% -----------    
end

function po = epsFind(a,b,q,F0,tou)
 % binary search for the initialization of epsr
  feps = @(e)((F0+e)^(1/2)*e^(-1/q)) ;
  
  while feps(a)>tou, a = a/2; end
  while feps(b)<tou, b = b*2; end
  
  while b-a>1e-12
    po = (a+b)/2;
    if feps(po) < tou, a = po;
    else, b = po ;
    end
  end
  po = (a+b)/2 ; 
end
