clc;
clear;
close;

% 数据预处理
%% 初始化
folder = 'D:\\Desktop';   %本文件夹存储位置
% 光谱波形数据文件
Wave_folder = sprintf('%s\\Data processing\\data\\wavelength144.mat',folder);
% 采集样品光谱数据文件
Black_White_Data_folder = sprintf('%s\\Data processing\\Result\\Black_White\\post_processing_data.xlsx',folder);
% 平滑数据保存文件
smooth_mat_file_name = sprintf('%s\\Data processing\\Result\\Smooth\\SSC\\Smooth_Results.mat', ...
                       folder);
smooth_file_name = sprintf('%s\\Data processing\\Result\\Smooth\\SSC\\Smooth_Results.tif', ...
                   folder);  
%% 加载数据
load(Wave_folder);                                     %波长数据读取
data = xlsread(Black_White_Data_folder);               %光谱数据读取
data =  data(1:120,:);
%% 数据预处理 SG+SNV+归一化

SG_data = SG(data,2,13); %SG滤波
MSC_data = MSC(SG_data);

% MSC_data = SG_data;
% SNV_data = SNV(MSC_data); %SNV
SNV_data = MSC_data;

% Zero_data = normalization(SNV_data,1); %数据归一化
Zero_data = SNV_data;

Post_smooth_data = Zero_data;    %用于下方数据更改方便

%% 预处理数据画图
fig_smooth = figure(1);             %创建图像窗口
subplot(2,2,1);              %4个子图像，在第1个窗口画图
plot(wavelength144,data);    %原始光谱
title(" source data");        %添加标题
subplot(2,2,2);              %4个子图像，在第2个窗口画图
plot(wavelength144,SG_data); %SG光谱
title(" SG data ");           %添加标题
subplot(2,2,3);              %4个子图像，在第3个窗口画图
plot(wavelength144,MSC_data); %SNV光谱
title(" SNV data ");       %添加标题
subplot(2,2,4);              %4个子图像，在第4个窗口画图
plot(wavelength144,SNV_data); %MSC光谱
title(" Std data ");       %添加标题
save(smooth_mat_file_name, 'Post_smooth_data')
saveas(fig_smooth, smooth_file_name, 'tiff');    %保存数据
