% Written by Ye Wang 03/2022(E-mail: w773664703@gmail.com)
% adaptively update strategy
% INPUT ::
% - weps: the old weight
% - Rk0: the old rank
% - Rk1: the new rank
% - rc: dhe dimension of a squared singular value matrix
% - nssgv: nonzeros of the new singular value
% - mu: 0 < mu < 1

% OUTPUT ::
% - weps: the new weight
function weps = update_eps(weps,Rk0,Rk1,rc,nssgv,mu)
% if (Rk1 == rc) || (Rk0 == Rk1)
%   weps = weps .* mu;
%   return
% end

if Rk0 >= Rk1
  weps(1:Rk1) = weps(1:Rk1) .* mu;
  tau_1 = nssgv + weps(Rk1);
  if Rk1 == rc
    return
  end
  tau_2 = weps(Rk1+1);
  mk0k1 = (tau_1 >= tau_2);
  if Rk0 == Rk1 
    mk1k1 = (weps(Rk1+1:end) < mu*tau_1);
    weps(Rk1+1:end) = mk0k1 .* weps(Rk1+1:end) + ...
    (~mk0k1) .* (mk1k1.*weps(Rk1+1:end) + (~mk1k1).*mu*tau_1);
    return
  else
    mk1k1 = (weps(Rk1+1:Rk0) < mu*tau_1);
    weps(Rk1+1:Rk0) =  mk0k1 .* weps(Rk1+1:Rk0) + ...
      (~mk0k1) .* (mk1k1.*weps(Rk1+1:Rk0) + (~mk1k1).*mu*tau_1);
    tau_3 = weps(Rk0);
    mk2k1 = (weps(Rk0+1:rc) < tau_3);
    weps(Rk0+1:rc) = (weps(Rk0+1:rc).*mk2k1) +  (tau_3.*(~mk2k1)) ;
  end
else
  tau_3 = weps(Rk0);
  weps(1:Rk0) = weps(1:Rk0) .* mu;

  mk0k1 = weps(Rk0+1:Rk1) < tau_3;
  weps(Rk0+1:Rk1) = mu .* ( mk0k1.*weps(Rk0+1:Rk1) + tau_3.*(~mk0k1));
  if Rk1 == rc
    return
  end
  tau_2 = weps(Rk1+1);
  tau_1 = weps(Rk1) + nssgv;
  mk1k1 = tau_1 > tau_2;
  mk2k1 = ( weps(Rk1+1:rc) > mu*tau_1 );
  weps(Rk1+1:rc) = mk1k1.*weps(Rk1+1:rc) + ...
    (~mk1k1).*( mk2k1.*weps(Rk1+1:rc) + (~mk2k1).*mu*tau_1);
end
end