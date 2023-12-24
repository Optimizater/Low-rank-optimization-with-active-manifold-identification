function [U0, S, V, output ] = IRNN( D, lambda, theta, para )
output.method = 'IRNN';
if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

objstep = 1;

maxIter = para.maxIter;
tol = para.tol*objstep;

regType = para.regType;
[row, col, data] = find(D);
[m, n] = size(D);


R = para.R;
U0 = para.U0;
U1 = U0;
Ui = U0;

[~, S, V0] = svd(U0'*D, 'econ');
V1 = V0;
Vi = V0;
sigma = diag(S);
spa = sparse(row, col, data, m, n);

clear D;

obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);

part0 = partXY(U0', V0', row, col, length(data));
part1 = partXY(U1', V1', row, col, length(data));

c = 1;

for i = 1:maxIter
    tt = cputime;
    
    % the first proximal operator
    [Ui, S, Vi, spa] = proxOperator_IRNN(sigma, U1, V1, U0, V0, spa, 0, part0, part1, 0, data, theta, lambda, c, regType, maxR);
    sigma = diag(S);
    
    
    
        
    
    U0 = U1;   
    U1 = Ui;
    V0 = V1;
    V1 = Vi;
    part0 = part1;
    

    part1 = partXY(Ui', Vi', row, col, length(data));
    
    objVal = (1/2)*sum((data - part1').^2);
    objVal = objVal + lambda*sum(1-exp(-theta*sigma)); % exponential
    
      
    
    
    
    c = c + 1;

    if(i > 1)
        delta = (obj(i - 1)- objVal)/objVal;
    else
        delta = inf;
    end
    pwIter = 0;
    
%     fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; power(%d,%d); acc:%d \n', ...
%         i, objVal, delta, nnz(S), lambda, pwIter, size(R, 2), c);
    
    Time(i) = cputime - tt;
    obj(i) = objVal;
    
    % testing performance
    if(isfield(para, 'test'))
        tempS = eye(size(U1,2), size(V1,2));
        if(para.test.m ~= m)
            RMSE(i) = MatCompRMSE(V1, U1, tempS, para.test.row, para.test.col, para.test.data);
        else
            RMSE(i) = MatCompRMSE(U1, V1, tempS, para.test.row, para.test.col, para.test.data);
        end
        fprintf('method: %s data: %s  RMSE %.2d \n', output.method, para.data, RMSE(i));
    end
    
    if(i > 1 && abs(delta) < tol)
        break;
    end
    
    if(sum(Time) > para.maxtime)
        break;
    end
end

output.obj = obj(1:i);
[U0, S, V] = svd(U1, 'econ');
V = V1*V;
output.Rank = nnz(S);
output.RMSE = RMSE(1:i);

Time = cumsum(Time);
output.Time = Time(1:i);
output.data = para.data;
end

function[Ui, S, Vi, spa] = proxOperator_IRNN(sigma, U1, V1, U0, V0, spa, bi, part0, part1,part12, data, theta, lambda, c, regType, maxR)
    a = (c+1)/(c+2);
    if bi == 0
        part0 = data - part1';
    else
        part0 = data - (1 + bi)*part1' + bi*part0'; % - a*(part12'-part1');
    end
    setSval(spa, part0, length(part0));
    
     
    
    
    w = theta*exp(-theta*sigma); % exponential
    
    [Ui, S, Vi] = accExactSVD_APGnc( U1, V1, U0, V0, spa, bi, maxR);
    sigma = diag(S);
    sigma = max(sigma-lambda*w,0);
    S = diag(sigma);
    %[ Ui, S, Vi ] = GSVT(hZ, lambda, theta, regType);
    Ui = (Ui*S);   
    
    
end

function [U, S, V] = accExactSVD_IRNN( U1, V1, U0, V0, bi, k)

m = size(U1,1);
n = size(V1,1);
Afunc  = @(x) ((1+bi)*(U1*(V1'*x)) - bi*(U0*(V0'*x)));
Atfunc = @(y) ((1+bi)*(V1*(U1'*y)) - bi*(V0*(U0'*y)));

rnk = min(min(m,n), ceil(1.25*k));
% rnk = k;
[U, S, V] = lansvd(Afunc, Atfunc, m, n, rnk, 'L');

U = U(:, 1:k);
V = V(:, 1:k);

S = diag(S);
S = S(1:k);
S = diag(S);

end

function [U, S, V] = accExactSVD_APGnc( U1, V1, U0, V0, spa, bi, k)

m = size(U1,1);
n = size(V1,1);
Afunc  = @(x) (spa*x + (1+bi)*(U1*(V1'*x)) - bi*(U0*(V0'*x)));
Atfunc = @(y) (spa'*y + (1+bi)*(V1*(U1'*y)) - bi*(V0*(U0'*y)));

rnk = min(min(m,n), ceil(1.25*k));
% rnk = k;
[U, S, V] = lansvd(Afunc, Atfunc, m, n, rnk, 'L');

U = U(:, 1:k);
V = V(:, 1:k);

S = diag(S);
S = S(1:k);
S = diag(S);

end

