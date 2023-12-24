% test IRNN for matrix completion applied to image recovery

clear;
clc;
% addpath(genpath(cd))
Xfull = double(imread('testimage.jpg'))/255;
% Xfull = double(imread('testimage.jpg')/255.0);
%%
[m,n,c]=size(Xfull);
pp = 0.3; % Sampling Rate- SR
p = ceil(pp * m* n); % 50% entries are missing

[I, J, col, omega] = myRandsample(m, n, p);
M = opRestriction(m*n,omega);
%%
% observed image
ind = zeros(m,n);
ind(omega) = 1 ;
mask(:,:,1)=ind;
mask(:,:,2)=ind;
mask(:,:,3)=ind;
Xmiss = Xfull.*mask; 
 
%% % choose penalty in IRNN
fun = 'lp' ;        gamma = 0.01;
% fun = 'scad' ;      gamma = 10;
% fun = 'logarithm' ; gamma = 0.1; % or 1
% fun = 'mcp' ;       gamma = 0.1;
% fun = 'etp' ;       gamma = 0.001;

lambda_rho = 0.5;
options.gamma = 1 ;
options.lambda_rho = 1e-2 ;
options.tol = 1e-3; 
for i = 1 : 3
    fprintf('chanel %d\n',i) ;
    X = Xfull(:,:,i);
    x = X(:) ;
    y = M(x,1) ;   
    options.lambda_Init = 0.1 ;
%     lambda_Init = max(abs(M(y,2)))*1000;
    SOL_IRNN = IRNN(fun,y,M,m,n,options);
%     IRNN(fun,y,M,m,n,gamma,lambda_Init,lambda_rho,tol)
end
%%
Xhat = max(Xhat,0);
Xhat = min(Xhat,255);
psnr = PSNR(Xfull,Xhat,max(Xfull(:)))

% figure(1)
% subplot(1,2,1)
% imshow(Xmiss/255);
% subplot(1,2,2)
% imshow(Xhat/255);



