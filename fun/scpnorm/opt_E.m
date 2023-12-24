function E_omega = opt_E(X,Y_omega,mv,opt)
    omega = opt.omega;
    D_omega = opt.D_omega;
    p = opt.p;
    [m,n] = size(omega);
    E_omega = zeros(size(omega));
    X_omega = X.*omega;
    H_omega = X_omega - D_omega - (1.0/mv)*Y_omega;

%self
    E_omega = (2.0/(mv + 2)) * H_omega;
    %E_omega = H_omega;
    
    
%Nie  
%     lambda = (1.0/mv);
%     v = (lambda*p*(1-p))^(1.0/(2-p));
%     v1 = v + lambda*p*(abs(v))^(p-1);
%     
%     [r,c] = find(omega);
%     for k=1:length(r)
%         if (H_omega(r(k),c(k)) >= -v1) && (H_omega(r(k),c(k)) <= v1)
%             E_omega(r(k),c(k)) = 0;
%         else
%             E_omega(r(k),c(k)) = GST(lambda,p,H_omega(r(k),c(k)));
%             if h(E_omega(r(k),c(k)),H_omega(r(k),c(k)),opt) > h(0,H_omega(r(k),c(k)),opt)
%                 E_omega(r(k),c(k)) = 0;
%             end
%         end
%     end  
end

