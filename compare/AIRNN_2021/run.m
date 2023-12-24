clear; clc;
clear; clc;
addpath('tools')
addpath('data');
dataset = {'movielens1m'};

for j = 1:1
    load(['data/',dataset{j},'.mat']);
    %%% arr = data;
    [row, col, val] = find(data);

    [m, n] = size(data);
    
    
    clear user item;
    

    val = val - mean(val);
    val = val/std(val);

    idx = randperm(length(val));

    traIdx = idx(1:floor(length(val)*0.5));
    tstIdx = idx(ceil(length(val)*0.5): end);

    clear idx;

    traData = sparse(row(traIdx), col(traIdx), val(traIdx));
    traData(size(data,1), size(data,2)) = 0;

    para.test.row  = row(tstIdx);
    para.test.col  = col(tstIdx);
    para.test.data = val(tstIdx);
    para.test.m = m;
    para.test.n = n;

    clear m n;
    clear data;
%%
    lambda = 2000;
    theta = sqrt(lambda);
    if j == 1
        para.maxR = 5;
        para.maxtime = 200;
    end
    if j == 2
        para.maxR = 8;
    end
    if j == 3
        para.maxR = 13;
    end
    if j == 4
        para.maxR = 9;
    end
    if j == 5
        para.maxR = 9;
    end
    para.regType = 4;
    para.maxIter = 10000;
    para.tol = 1e-5;
    [m, n] = size(traData);
    R = randn(n, para.maxR); % 
    para.R = R;
    para.data = dataset{j};
    clear m n;
    U0 = powerMethod( traData, R, para.maxR, 1e-6);
    
    para.U0 = U0;

%% IRNN
     method = 1;
     [~, ~, ~, out{method}] = IRNN( traData, lambda, theta, para );
 
    % % our AIRNN 
     method = 2;
%      [~, ~, ~, out{method}] = AccIRNN( traData, lambda, 0.1, para );
    [~, ~, ~, out{method}] = AccIRNN( traData, 200, 0.1, para );
%     % 
 
    close all;
    plot(out{1}.Time, log(out{1}.obj), 'k');
    hold on;
    
    plot(out{2}.Time, log(out{2}.obj), 'r');
    hold on;

     legend('IRNN', 'AIRNN');

    
end