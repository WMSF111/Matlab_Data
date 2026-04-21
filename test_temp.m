clc; clear; close all;
clear functions;
project_root = fileparts(mfilename('fullpath'));
lang_mode = 'zh';
normalize_mode = 'none';
addpath(genpath(fullfile(project_root, 'Package')));
addpath(genpath(fullfile(project_root, 'Time_Prediction')));

e_nose_data_root = fullfile(project_root, 'data', 'E_nose');
e_nose_feature_modes = {'med','mode'};
e_nose_use_baseline_removed = false;

result70 = predict_property_time_series(70, [], [], lang_mode, normalize_mode);
result80 = predict_property_time_series(80, [], [], lang_mode, normalize_mode);
result90 = predict_property_time_series(90, [], [], lang_mode, normalize_mode);
result_compare = compare_temperature_time_series([70 80 90], 0:20, [], lang_mode, normalize_mode);

% r70 = plot_e_nose_time_series(70, e_nose_feature_modes, e_nose_data_root, lang_mode, normalize_mode, e_nose_use_baseline_removed);
% r80 = plot_e_nose_time_series(80, e_nose_feature_modes, e_nose_data_root, lang_mode, normalize_mode, e_nose_use_baseline_removed);
% r90 = plot_e_nose_time_series(90, e_nose_feature_modes, e_nose_data_root, lang_mode, normalize_mode, e_nose_use_baseline_removed);
% rc  = compare_e_nose_time_series([70 80 90], e_nose_feature_modes, e_nose_data_root, lang_mode, normalize_mode, e_nose_use_baseline_removed);

% PATH = 'D:\HXR\Matlab\data\E_nose\90\2\奇数\15.csv';
% % 读取 CSV 文件
% data = readtable(PATH);
% 
% % 对 MQ138 列加 100
% data.MQ135 = data.MQ135 - 100;
% data.TGS2612 = data.TGS2612 - 100;
% 
% % 保存为 CSV 文件
% writetable(data, PATH);
% % % 读取 CSV 文件
% data = readtable(PATH);
% 
% % 获取所有列名
% vars = data.Properties.VariableNames;
% 
% % 对每一列数据加 100（假设都是数值列）
% for i = 1:length(vars)
%     if isnumeric(data.(vars{i}))
%         data.(vars{i}) = data.(vars{i}) + 100;
%     end
% end
% 
% % 保存为新的 CSV 文件
% writetable(data, PATH);