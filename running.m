clc;clear;close all;
cd('E:\study\algriothm\matlab\Data processing');
clear functions;

% 1) 查看 all_csv_data.csv 可用属性列
run_property_prediction('list');

% 2) 普通建模示例：预测 a*，方法 SPA，预处理 SG+MSC+SNV
run_property_prediction('a*','spa','sg+msc+snv',3,15);

% % 3) 普通建模示例：预测 Mrix，方法 PCA，预处理 SG+MSC
% run_property_prediction('Mrix','pca','sg+msc',3,15);

% % 4) 特征筛选示例：预测 a*，启用特征筛选流程（corr_topk）
% run_property_prediction('a*','fs','sg+msc+snv',3,15);