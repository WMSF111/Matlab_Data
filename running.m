% 主入口脚本：在这里统一配置预处理、特征选择和泛化增强开关。
clc; clear; close all;
clear functions;

% 预测目标、预处理方式和候选特征方法。
property_name = 'a*';
filter_method = 'sg+msc+snv';
feature_selection_method = {'cars'};
sg_orders = [3];
sg_windows = [5];

% 先读取默认开关，再覆盖成一套偏稳健、抗过拟合的推荐配置。
generalization_options = default_generalization_options();
% 统一降维上限：
% - post_feature_pca_components ：非降维模型前置 PCA 投影维数
% - pls_max_lv_cap：PLS 最多搜索的潜变量数
% - pcr_max_pc_cap：PCR 最多搜索的主成分数
% 以后如果想统一改最终降维强度，只改这一行即可。
shared_component_cap = 40;
% 抗过拟合推荐配置：
% 嵌套交叉验证选特征、稳定特征筛选、独立外部验证、模型简化、
% CNN 正则化，以及 XGBoost/CNN 的快速参数网格搜索。
generalization_options.feature.nested_cv_selection = true;
generalization_options.feature.use_stable_features = true;
generalization_options.feature.stability_threshold = 0.65;
% 先用 CARS/PCA/SPA 等方法筛一批特征，再用 PCA 主成分投影压到更低维。
% 这个开关不会改写 feature_selection_method，只是在筛选之后追加降维。
generalization_options.feature.post_feature_pca_projection = true;
generalization_options.feature.post_feature_pca_components = shared_component_cap;

generalization_options.data_processing.data_augmentation = true;
generalization_options.data_processing.augmentation_copies = 1;
generalization_options.data_processing.noise_std = 0.002;
generalization_options.data_processing.max_shift = 1;

generalization_options.validation.use_external_holdout = true;
generalization_options.validation.external_ratio = 0.20;

generalization_options.model.simplify_pls_pcr = true;
generalization_options.model.pls_max_lv_cap = shared_component_cap;
generalization_options.model.pcr_max_pc_cap = shared_component_cap;
generalization_options.model.xgboost_simplify = true;
generalization_options.model.cnn_simplify = true;
generalization_options.model.l1_l2_regularization = true;
generalization_options.model.l2_lambda = 1e-4;
generalization_options.model.dropout = true;
generalization_options.model.dropout_rate = 0.30;
generalization_options.model.early_stopping = true;
generalization_options.model.validation_patience = 8;
generalization_options.model.xgboost_grid_mode = 'quick';
generalization_options.model.cnn_grid_mode = 'quick';
generalization_options.evaluation.enforce_small_rc_rp_gap = true;
generalization_options.evaluation.rc_rp_gap_threshold = 0.05;
% true=显示并保存滤波图/特征筛选图；false=只显示，停留后自动关闭，不保存。
generalization_options.evaluation.save_smooth_and_feature_plots = true;

% 如果想直接在入口脚本里自定义 XGBoost 参数网格，可以在这里填写。
% 只要下面四组数组都非空，主流程就会优先使用这些自定义参数，
% 不再走 compare_property_prediction_pipeline.m 里的 off/quick/full 固定网格。
generalization_options.model.xgb_num_learning_cycles = [80 120 150];
generalization_options.model.xgb_learn_rates = [0.03 0.05];
generalization_options.model.xgb_max_num_splits = [7 10];
generalization_options.model.xgb_min_leaf_sizes = [12 16 20];

% 如果想直接在入口脚本里自定义 CNN 参数网格，也可以在这里填写。
% 只要下面六组配置都非空，主流程就会优先使用这些自定义参数。
generalization_options.model.cnn_conv_channel_sets = {[16 32], [32 64], [16 32 64]};
generalization_options.model.cnn_fc_units_grid = [48 64 96];
generalization_options.model.cnn_dropout_rates = [0.10 0.20];
generalization_options.model.cnn_max_epochs_grid = [150];
generalization_options.model.cnn_mini_batch_sizes = [8 16];
generalization_options.model.cnn_initial_learn_rates = [1e-3 2e-3];

% 可选：更重的系统搜索，适合最终精调。
% generalization_options.model.xgboost_grid_mode = 'full';
% generalization_options.model.cnn_grid_mode = 'full';
% 可选：最快速的基线，只跑单组参数。
% generalization_options.model.xgboost_grid_mode = 'off';
% generalization_options.model.cnn_grid_mode = 'off';

% 如果想恢复使用内置 off/quick/full 网格，把上面的自定义数组清空即可。
% generalization_options.model.xgb_num_learning_cycles = [];
% generalization_options.model.xgb_learn_rates = [];
% generalization_options.model.xgb_max_num_splits = [];
% generalization_options.model.xgb_min_leaf_sizes = [];
% generalization_options.model.cnn_conv_channel_sets = {};
% generalization_options.model.cnn_fc_units_grid = [];
% generalization_options.model.cnn_dropout_rates = [];
% generalization_options.model.cnn_max_epochs_grid = [];
% generalization_options.model.cnn_mini_batch_sizes = [];
% generalization_options.model.cnn_initial_learn_rates = [];

% SG 参数网格遍历；可以在这里扩展多个阶数和窗口长度组合。
for i = 1:numel(sg_orders)
    current_order = sg_orders(i);
    for j = 1:numel(sg_windows)
        current_window = sg_windows(j);
        % SG 窗口必须为奇数，且通常应大于多项式阶数。
        if mod(current_window, 2) == 0
            continue;
        end
        if current_window <= current_order
            continue;
        end
        fprintf('\n================ SG 参数网格测试开始 ================\n');
        summary = compare_property_prediction_pipeline(property_name, filter_method, feature_selection_method, current_order, current_window, false, generalization_options); %#ok<NASGU>
    end
end

% 合并历史汇总结果，方便比较不同运行批次。
merge_summary_csv_by_property('brix');

