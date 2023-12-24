function X = opt_X(E_omega,Y_omega,W,Z,mv,opt)
    D_omega = opt.D_omega;
    omega = opt.omega;
    lambda = opt.lambda;
    omega_ = ones(size(omega)) - omega;
    K_omega = E_omega + D_omega + (1.0/mv)*Y_omega;
    N = W - (1.0/mv)*Z;
    %N = W - lambda*Z;
    X_omega = (K_omega + N.*omega)/2.0;
    X_omega_ = N.*omega_;
    X = X_omega + X_omega_;
end

