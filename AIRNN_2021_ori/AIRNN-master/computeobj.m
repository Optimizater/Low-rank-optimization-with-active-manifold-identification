function[objVal] = computeobj(data,part1,lambda,theta,sigma,regType)
    if(regType == 4)
        objVal = (1/2)*sum((data - part1').^2);
        objVal = objVal + lambda*norm(sigma,theta)^(theta); 
%         objVal = objVal + lambda*sum(1-exp(-theta*sigma)); % exponential
    end
end