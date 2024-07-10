function W = opt_W(X,Z,mv,opt)
    lambda = opt.lambda;
    G = X + (1.0/mv)*Z;
    %G = X + lambda*Z;
    [Q,Sigma,R] = svd(G);
    p = opt.p;
    [m,n] = size(G);
    sigma = diag(Sigma);
    delta = zeros(size(sigma));
%      
%% self    
    tau = opt.tau;
    v = (2*lambda*(1-p))^(1.0/(2-p));
    v1 = v + lambda*p*v^(p-1);
    mn = min(m,n);
    for i = 1:mn
        s = sigma(i);
        if s >= v1
            x_ = GST(lambda,p,s);
        else
            x_ = 0;
        end
        tau_ = ((1.0/(2*lambda))*(x_-s)^2 + x_^p)^(1.0/p);
        if tau <= tau_
            delta(i) = s;
        else
            delta(i) = x_;
        end
    end
    
    Delta = zeros(size(Sigma));
    Delta(1:mn , 1:mn) = diag(delta);
    W = Q*Delta*R';
end

