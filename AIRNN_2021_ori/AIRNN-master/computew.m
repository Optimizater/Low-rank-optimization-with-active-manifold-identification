function[w] = computew(theta,sigma,regType)
    if(regType == 4)
%         w = theta*exp(-theta*sigma);
      w = theta.*(sigma).^(theta-1);
    end
    if(regType == 2)
        w = 1./(theta + sigma);
    end
end