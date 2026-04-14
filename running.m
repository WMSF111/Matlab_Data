clc; clear; close all;
clear functions;

% 一键运行：
% compare_property_prediction_pipeline(预测对象, 滤波方式, 特征选择方式)
% 例如：
%   compare_property_prediction_pipeline('a*', 'sg+msc+snv', 'cars')
% 若后两个参数不写，则默认使用：'sg+msc+snv' 和 'cars'
% summary = compare_property_prediction_pipeline('C*','sg+msc+snv','cars');
% summary = compare_property_prediction_pipeline('b*','sg+msc+snv','cars');
% 
% summary = compare_property_prediction_pipeline('a*','sg+msc+snv','cars');
% summary = compare_property_prediction_pipeline('ho','sg+msc+snv','cars');
% 
% summary = compare_property_prediction_pipeline('L*','sg+msc+snv','cars' );

% summary = compare_property_prediction_pipeline('C*','sg+msc+snv','pca');
% summary = compare_property_prediction_pipeline('b*','sg+msc+snv','pca');

% summary = compare_property_prediction_pipeline('a*','sg+msc+snv','pca');
% summary = compare_property_prediction_pipeline('ho','sg+msc+snv','pca');

% summary = compare_property_prediction_pipeline('L*','sg+msc+snv','pca' );

% summary = compare_property_prediction_pipeline('C*','sg+msc+snv','corr_topk');
% summary = compare_property_prediction_pipeline('b*','sg+msc+snv','corr_topk');

% summary = compare_property_prediction_pipeline('a*','sg+msc+snv','corr_topk');
% summary = compare_property_prediction_pipeline('ho','sg+msc+snv','corr_topk');

% summary = compare_property_prediction_pipeline('L*','sg+msc+snv','corr_topk' );

% summary = compare_property_prediction_pipeline('C*','sg+msc+snv','spa');
% summary = compare_property_prediction_pipeline('b*','sg+msc+snv','spa');

% summary = compare_property_prediction_pipeline('a*','sg+msc+snv','spa');
% summary = compare_property_prediction_pipeline('ho','sg+msc+snv','spa');

% summary = compare_property_prediction_pipeline('L*','sg+msc+snv','spa' );
% % 1) 将 Result\Model 下所有运行目录中的所有回归器图像分别拼成大图
% run_list = { ...
%     'Run_20260414_122245', ...
%     'Run_20260414_122631', ...
%     'Run_20260414_122947', ...
%     'Run_20260414_123328', ...
%     'Run_20260414_123717', ...
%     'Run_20260414_155726', ...
%     'Run_20260414_160039', ...
%     'Run_20260414_160351', ...
%     'Run_20260414_160728', ...
%     'Run_20260414_161102', ...
%     'Run_20260414_161410', ...
%     'Run_20260414_161645', ...
%     'Run_20260414_161929', ...
%     'Run_20260414_162210', ...
%     'Run_20260414_162502', ...
%     'Run_20260414_162800', ...
%     'Run_20260414_163037', ...
%     'Run_20260414_163305', ...
%     'Run_20260414_163519', ...
%     'Run_20260414_163752'};
% 
% model_list = {'PLS', 'PCR', 'SVR', 'RF', 'GPR', 'KNN'};
% 
% for i = 1:numel(run_list)
%     for j = 1:numel(model_list)
%         merge_model_plots(run_list{i}, model_list{j});
%     end
% end
merge_summary_csv_by_property('a*');
merge_summary_csv_by_property('b*');
merge_summary_csv_by_property('C*');
merge_summary_csv_by_property('L*');
merge_summary_csv_by_property('ho');

