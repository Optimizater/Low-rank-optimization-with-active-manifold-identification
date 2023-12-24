function weps = update_eps(weps,Rk0,Rk1,rc,nssgv,mu)
% adaptively update strategy

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
  mk1k1 = (weps(Rk1+1:Rk0) < mu*tau_1);
  weps(Rk1+1:Rk0) =  mk0k1 .* weps(Rk1+1:Rk0) + ...
    (~mk0k1) .* (mk1k1.*weps(Rk1+1:Rk0) + (~mk1k1).*mu*tau_1);
  tau_3 = weps(Rk0);
  mk2k1 = (weps(Rk0+1:rc) < tau_3);
  weps(Rk0+1:rc) = (weps(Rk0+1:rc).*mk2k1) +  (tau_3.*(~mk2k1)) ;
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