function y = lp_sg(x,p,lambda)
% supergradient of lp penalty

x = abs(x) ;
epsilon = 1e-2 ;
y = lambda*p*(x+epsilon).^(p-1) ; % 
% y = lambda*(x+epsilon).^(p-1) ;
