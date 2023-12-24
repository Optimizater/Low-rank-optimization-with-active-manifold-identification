%% the result of x_*
function T = GST(lambda,p,sigma)
%     v = (2*lambda*(1-p))^(1.0/(2-p));
%     v1 = v + lambda*p*v^(p-1);
    J = 20;
    x_k = sigma;
%     if sigma > 0
%         sgn = 1;
%     else
%         sgn = -1;
%     end
    for k=1:J
        x_k1 = sigma - lambda*p*(x_k)^(p-1);
        x_k = x_k1;
    end
    T = x_k;
end

