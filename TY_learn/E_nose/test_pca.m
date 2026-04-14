%% ========================
%% 电子鼻 时序PCA降维 + 时间趋势曲线
%% 每个文件：时序数据 → PCA → 画一条连续PCA线
%% ========================
clear; clc; close all;

% 修复字体设置错误
set(0,'DefaultAxesFontName','SimHei');
set(0,'DefaultTextFontName','SimHei');

%% 1. 你的文件夹
folder = '2026_3_24_2\偶数';

%% 2. 读取所有CSV
fileList = dir(fullfile(folder, '*.csv'));
if isempty(fileList)
    error('未找到CSV文件！');
end

%% 3. 遍历每个文件 → 时序PCA → 画线
figure('Color','w','Position',[200,200,700,600]);
hold on; grid on;
title('时序PCA降维曲线（按时间变化趋势）','FontSize',14);
xlabel('PC1');
ylabel('PC2');

colors = lines(length(fileList)); 

for i = 1:length(fileList)
    fname = fullfile(folder, fileList(i).name);
    fprintf('PCA处理：%s\n', fileList(i).name);
    
    % 读取数据
    T = readtable(fname);
    T(:,1) = [];          % 删除第一列无用数据
    data = table2array(T);
    
    % 滤波（平滑去噪）
    for c = 1:size(data,2)
        data(:,c) = movmean(data(:,c), 5);
    end
    
    % ======================
    % 时序PCA（每一行 = 一个时间点）
    % ======================
    data_z = zscore(data); 
    [coeff, score, ~, ~, explained] = pca(data_z);
    
    % ======================
    % 绘制：时间序列的PCA连续线
    % ======================
    plot(score(:,1), score(:,2), ...
        'o-', 'LineWidth',1.8, 'Color',colors(i,:), 'DisplayName', fileList(i).name);
end

legend('Location','best');
fprintf('\n[OK] 所有时序PCA曲线绘制完成！\n');