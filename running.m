clc; clear; close all;
clear functions;

% % SG 双参数网格测试入口
property_name = 'b*';
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
