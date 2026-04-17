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

% 
% % 运行结束后，显示并保存 all_csv_data.csv 的 RGB 总图
% show_all_csv_rgb_grid;

clc; clear; close all;
clear functions;

% % SG 双参数网格测试入口
property_name = 'Brix';
filter_method = 'sg';
% feature_selection_method = {'cars','pca', 'corr_topk', 'spa'};
feature_selection_method = {'cars'};
sg_orders = [3];
sg_windows = [5];


for i = 1:numel(sg_orders)
    current_order = sg_orders(i);
    for j = 1:numel(sg_windows)
        current_window = sg_windows(j);

        % SG 窗口必须为奇数，且通常应大于阶数
        if mod(current_window, 2) == 0
            continue;
        end
        if current_window <= current_order
            continue;
        end

        fprintf('\n================ SG 参数网格测试开始 ================\n');
        % fprintf('预测对象：%s | 滤波方式：%s | 特征筛选：%s | SG阶数=%d | SG窗口=%d\n', ...
        %     property_name, filter_method, feature_selection_method, current_order, current_window);

        summary = compare_property_prediction_pipeline(property_name, filter_method, feature_selection_method, current_order, current_window); 
    end
end


merge_summary_csv_by_property('brix');


% % 运行结束后，显示并保存 all_csv_data.csv 的 RGB 总图
% show_all_csv_rgb_grid;
