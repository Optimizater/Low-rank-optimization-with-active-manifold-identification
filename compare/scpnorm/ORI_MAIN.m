% Guang Guo, Northeastern University, June 2019. 
% Contact information: see readme.txt.
%
% Reference: 
% Li Guorui & Guo Guang & Peng Sancheng; Wang, Cong; Yu, Shui. Matrix Completion via
% Schatten Capped p Norm.
% 
% First written by guangguo, Northeastern Universiy, June 2019.

%%
clear;
clc;
rng('default');
cwd = cd;
path = strcat(cwd,'\image\lena.png');
img_data = double(imread(path))/255;
img_size = size(img_data);
T_flag = 0;%Is transposition,0 isn't transposed.
if img_size(1) < img_size(2)%if m < n, transposition
    img_ori = permute(img_data,[2,1,3]);
    T_flag = 1;
    img_size = [img_size(2),img_size(1),img_size(3)];
else
    img_ori = img_data;
end
m = img_size(1);
n = img_size(2);

%% strictly low rank
for channel = 1:3 
    img = img_ori(:,:,channel);
    [U,S,V] = svd(img);
    rank = 20; %strictly rank
    for j = rank:min(m,n)
        S(j,j) = 0;
    end
    img_ori(:,:,channel) = U*S*V';
end

%% block mask
mask_path = strcat(cwd,'\mask\block_column.bmp');
mask = double(imread(mask_path))/255.0;
mask = imresize(mask,img_size(1:2));
omega = zeros(size(mask));
omega_temp = mask(:,:,1);
for channel = 1:3
    omega(:,:,channel) = omega_temp;
end

%% random mask
lost = 0.5;%random mask rate
rnd_idx = randi([0, 100-1], m, n);
old_idx = rnd_idx;
lost = lost * 100;
fprintf('loss: %d%% elements are missing.\n', lost);
rnd_idx = double(old_idx < (100-lost));
omega = rnd_idx; % index matrix of the known elements
omega_temp = omega;
for channel = 1:3
    omega(:,:,channel) = omega_temp;
end

omega_img = omega .* img_ori;%input D with loss data

max_iter = 100;%iteration times
opt.lambda = 1;
opt.tau = 30;
opt.p =0.01;%value of p

fro = zeros(100,1);%mse
peak_snr = zeros(100,1);%peak_snr
mean_snr = zeros(100,1);%snr

for i = 1:100
    img_new = zeros(size(img_ori)); % the result of completed data
    t_0 = 1;
    X_0 = zeros(img_size);
    A_0 = X_0;
    for channel = 1:3
        opt.omega = omega(:,:,channel);
        opt.D_omega = omega_img(:,:,channel);
        Y_omega = opt.D_omega;
        E_omega = opt.D_omega;
        W = opt.D_omega;
        Z = Y_omega;
        rou = 1.5;
        mv = 1.5;

        for iter = 1:max_iter
            X = opt_X(E_omega,Y_omega,W,Z,mv,opt);
            E_omega = opt_E(X,Y_omega,mv,opt);
            W = opt_W(X,Z,mv,opt);
            Y_omega = Y_omega + mv*(E_omega - X .* opt.omega + opt.D_omega); 
            Z = Z + mv*(X - W);
            mv = rou*mv;
        end
        img_new(:,:,channel) = X;
    end
    [peaksnr,snr] = psnr(img_new,img_ori);
   
    fro_norm = norm(img_ori(:) - img_new(:), 2)/norm(img_ori(:), 2);
    fprintf('p: %2.2f, ', opt.p);
    fprintf('fro_norm: %f, ', fro_norm);
    fprintf('peaksnr: %f, ', peaksnr);
    fprintf('snr: %f.\n', snr);
    
    fro(i) = fro_norm;
    peak_snr(i) = peaksnr;
    mean_snr(i) = snr;   
    
    opt.p = 0.01+opt.p;
end
%% plot
% [~,index] = max(peak_snr);
opt.p = 0.2;
opt.lambda = 1;
opt.tau = 30;
max_iter = 1e3;



for channel = 1:3
%     opt.p = index/100.0;
    opt.omega = mask;
    opt.D_omega = XM(:,:,channel);
    Y_omega = opt.D_omega;
    E_omega = opt.D_omega;
    W = opt.D_omega;
    Z = Y_omega;
    rou = 1.5;
    mv = 1.5;

    for iter = 1:max_iter
        X = opt_X(E_omega,Y_omega,W,Z,mv,opt);
        E_omega = opt_E(X,Y_omega,mv,opt);
        W = opt_W(X,Z,mv,opt);
        Y_omega = Y_omega + mv*(E_omega - X .* opt.omega + opt.D_omega); 
        Z = Z + mv*(X - W);
        mv = rou*mv;
    end
    img_new(:,:,channel) = X;
end

% fprintf('the best value of p is: %f, ', opt.p);
%%
figure(1)
subplot(131);
imshow(img_ori);
title('ori image');

subplot(132);
imshow(XM);
title('mask image');

subplot(133);
imshow(img_new);
title('completed image');

%%
figure(2);
[hAx,hline1,hline2] = plotyy(0.01:0.01:1,peak_snr,0.01:0.01:1,fro);%,1:100,mean_snr);
xlabel('p');
ylabel(hAx(1),'PSNR'); % left y-axis
ylabel(hAx(2),'RE'); % right y-axis

% hLine1.LineStyle = ':';
% hLine2.LineStyle = '-.';
% hLine1.Color = 'black';
% hLine2.Color = 'black';
% hline1.LineWidth = 2;
% hline2.LineWidth = 2;
set(hline1,'linestyle',':','linewidth',2);
set(hline2,'linestyle','--','linewidth',2);
% set(hAx(1),'YColor','black');
% set(hAx(2),'YColor','black');
ylim([4,56]);
%text(61,35.77,'*','FontSize',30);
set(gca,'ytick',(4:5:56));
legend([hline1,hline2],{'PSNR';'RE'});
%grid minor;
%grid on;