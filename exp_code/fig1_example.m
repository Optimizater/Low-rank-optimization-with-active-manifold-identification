%% ------------ rank-identification Example ------------ 
clc, clear, format long; rng(22);
% ------------------------ init ------------------------ 
mat_size = [15 15];
Rank_range = [3, 5, 10];

% ----------------------- matrix -----------------------
nr = mat_size(1); nc = mat_size(2);
r = Rank_range(1); % the rank of matrix Y
missrate = 0.5; 

YB = rand(nr,r); YC = rand(r,nc); Y = YB * YC;
Y = Y./(10*min(min(Y)));

init_rank = 10; % the rank of first iteration matrix
sr = init_rank;
X0 = rand(nr,nc); [uY,sY,vY] = svd(X0);
X0 = uY(:,1:sr) * sY(1:sr,1:sr) * vY(:,1:sr)'; % low-rank initialization

% ----------------------- random mask -----------------------
M_org = zeros(nr,nc);
for i=1:nc
  randidx=randperm(nr,nr); % random sequence
  M_org(randidx(1:ceil(nr*missrate)),i)=1;
end
mask = ~M_org; 
Xm = Y.*mask; % sampling matrix

% IRNRI algorithm 
% --------------------- parameters --------------------- 
lambda = 1e-1;
itmax = 1e5; 
sp = 0.5; 
tol = 1e-7; 
klopt = 1e-5;
weps = 1e-16; % a small eps

options.max_iter = itmax; 
options.KLopt = klopt;
options.beta = 1.3; 
options.teps = weps;

times = 100;

optionsA = options;
AIR = IRNRI(X0, Xm, sp, lambda, mask, tol, optionsA);
%% use cluster point as optimal soltion to calculate the relative error
% optionsA.Rel = AIR.Xsol;
% AIR = IRNRI(X0, Xm, sp, lambda, mask, tol, optionsA); 

%% the trajectory for algorithm IRNRI
% figure(1)
% subplot(3, 1, 1)
% plot(AIR.rank)
% ylabel("rank of iteration")
% 
% subplot(3, 1, 2)
% plot(AIR.f)
% ylabel("objective")
% 
% subplot(3, 1, 3)
% plot(log10(AIR.RelDist))
% ylabel("$\log_{10}$ RelDist",'Interpreter','latex')

% plot into one figure
figure(2)
set(gcf, 'Position', [600 100 500 500]); 
yyaxis left
plot(AIR.rank, '-r'); hold on;
plot([1 length(AIR.rank)], [r r], 's-k');
ylabel("rank")

yyaxis right
plot(log10(AIR.RelDist), '-.b')
% plot(log10(AIR.RelErr), '-.r')
% axis([  ]) 
% yticks([-2 0 2 4 6 8 10 12])
% yticklabels({'-2','0','2', '4','6','8', '-2','0','2', })
ylabel("$\log_{10}$ RelDist",'Interpreter','latex')
xlabel("number of iterations")
hold off;
legend("rank of iteration", "rank at optimum", "log RelDist")
