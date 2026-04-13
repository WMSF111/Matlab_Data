clc; clear; close all;
clear functions;

% 一键运行参数
msc_ref_mode = 'first';      % 可选: 'mean' / 'median' / 'first'
snv_mode = 'robust';         % 可选: 'standard' / 'robust'

% 一键运行：
% 1) 分组1：SG+MSC+SNV
% 2) 分组2：SG+MSC+SNV + CARS筛选
% 3) 比较回归器：PLS / PCR / SVR / RF
% 4) 完整跑完后统一保存报告
summary = compare_a_prediction_pipeline(msc_ref_mode, snv_mode);
