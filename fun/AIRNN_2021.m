% AIRNN_2021 (X0,M,sp, lambda, mask, tol, options)
% matlab code AIRNN_2021 by calling IRNN_2021_ori(source code)
function Par = AIRNN_2021(X0,Xm,sp, lambda, mask, tol, options)
[m,n] = size(X0);

if(isfield(options, 'maxR')), maxR = options.maxR;
else,maxR = min(size(X0));
end

if(isfield(options, 'max_iter')),max_iter = options.max_iter;
else,max_iter = 2e3;
end

if(isfield(options, 'maxtime')),maxtime = options.maxtime;
else,maxtime = 20;
end

if(isfield(options, 'regType')),regType = options.regType;
else,regType = 4;
end

Xtesst = X0 .* mask;

para.maxtime = maxtime;

para.maxIter = max_iter;
para.regType = regType;
para.tol = tol;
para.maxR = maxR;

R = randn(n, para.maxR);
para.R = R;
para.data = Xm;
U0 = powerMethod(Xm, R, para.maxR, 1e-6);
para.U0 = U0;


%   para.test.data = options.Rel;
if isfield(options, 'Rel')
    para.Rel = options.Rel;
end

[Usol,Ssol , Vsol, out] = AIRNN( Xm, lambda, sp, para);

Par.Xsol =  Usol * Ssol * Vsol';
Par.RelErr = out.RMSE;
Par.iterTol = out.iterTol;
end