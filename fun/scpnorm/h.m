function result = h(x,a,opt)
    lambda = opt.lambda;
    p = opt.p;
    result = 0.5*(x-a)^2 + lambda*(abs(x))^p;
end

