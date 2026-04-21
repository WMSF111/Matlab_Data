function opts = default_generalization_options()
% 默认泛化增强开关
% 作用：
%   统一集中管理 compare_property_prediction_pipeline 的全部泛化相关参数。
%   这些参数不是某一个模型单独使用，而是贯穿“特征处理 -> 数据处理 ->
%   验证策略 -> 模型复杂度 -> 结果评估”整条训练链。
%
% 重要说明：
%   特征方法只由 feature_selection_method 决定。
%   也就是说：
%   - feature_selection_method='pca'，就走 pca
%   - feature_selection_method='cars'，就走 cars
%   - 不再使用额外的策略开关去改写特征方法
%
% 使用方式示例：
%   opts = default_generalization_options();
%   opts.feature.nested_cv_selection = true;
%   opts.validation.use_external_holdout = true;
%   opts.model.xgboost_simplify = true;

opts = struct();

% =========================
% 特征层配置 feature
% =========================
% nested_cv_selection
%   是否启用嵌套交叉验证特征选择。
%   开启后，特征选择只在训练集内部做内层交叉验证，
%   用来减少特征泄漏和结果乐观偏差。
%
% inner_cv_folds
%   嵌套交叉验证内层折数。
%
% use_stable_features
%   是否只保留在多个 fold 中都稳定出现的特征。
%
% stability_threshold
%   稳定特征出现频率阈值。
%   例如 0.60 表示至少在 60% 的 fold 中被选中。
opts.feature = struct( ...
    'nested_cv_selection', false, ...
    'inner_cv_folds', 5, ...
    'use_stable_features', false, ...
    'stability_threshold', 0.60, ...
    'post_feature_pca_projection', false, ...
    'post_feature_pca_components', 20);

% =========================
% 数据层配置 data_processing
% =========================
% data_augmentation
%   是否启用训练集增强。
%
% augmentation_copies
%   每个训练样本额外生成多少份增强样本。
%
% noise_std
%   增强时高斯噪声强度系数。
%
% max_shift
%   增强时光谱随机平移的最大步数。
opts.data_processing = struct( ...
    'data_augmentation', false, ...
    'augmentation_copies', 1, ...
    'noise_std', 0.003, ...
    'max_shift', 2);

% =========================
% 验证层配置 validation
% =========================
% use_external_holdout：是否保留独立外部验证集（不参与训练，参数搜索和特征选择）。
% external_ratio：外部验证集占比。
% random_seed：划分外部验证集时使用的随机种子。
opts.validation = struct( ...
    'use_external_holdout', false, ...
    'external_ratio', 0.20, ...
    'random_seed', 42);

% =========================
% 模型层配置 model
% =========================
% simplify_pls_pcr
%   是否限制 PLS/PCR 的复杂度。PLS 最多能用多少个潜变量。
%
% pls_max_lv_cap / pcr_max_pc_cap
%   PLS 潜变量数 / PCR 主成分数上限。PCR 最多能用多少个主成分
%
% l1_l2_regularization
%   当前主要对应神经网络的 L2 正则控制。限制模型权重不要太大、不要学得太激进的，主要作用是减少过拟合。
%
% l2_lambda  主要作用于：CNN
%   L2 正则强度。你希望多大程度去惩罚过大的权重。1e-5：很轻，1e-4：常用，1e-3：较强
%
% dropout / dropout_rate ：CNN
%   是否启用 Dropout 及其比例。每次训练时随机“丢掉”多少比例的神经元输出，来减少过拟合。0.1 ~ 0.3：比较常用，太小：效果弱，太大：模型可能学不动
%
% early_stopping / validation_patience，给 CNN 用的。如果开了，训练过程中会一直看验证集表现。一旦验证集不再变好，就提前结束训练。
%   是否启用早停及其耐心轮数。
%
% xgboost_simplify
%   是否简化 XGBoost 风格模型。防止树模型太复杂，把训练集拟合得过头。
%
% xgb_max_depth_cap / xgb_learn_rate_cap / xgb_min_leaf_floor
%   XGBoost 风格模型的复杂度控制参数。
% xgb_max_depth_cap：树深度上限，树越深，模型越复杂，越容易过拟合。越大：模型越灵活，越小：模型越保守
% xgb_learn_rate_cap：学习率上限，学习率越大，模型每棵树学得越激进，越容易过拟合。越大：模型学得快，越小：模型学得慢但可能更稳健
% xgb_min_leaf_floor：叶子节点最小样本数下限，越大：模型越保守，越小：模型越激进
%
% xgboost_grid_mode
%   XGBoost 网格模式：off / quick / full。只跑单组参数、跑一个小网格、跑一个更大的系统搜索网格。
%
% cnn_simplify
%   是否简化 CNN。
%
% cnn_max_blocks / cnn_fc_units_cap
%   CNN 卷积块数、全连接层宽度上限。
%
% cnn_grid_mode
%   CNN 网格模式：off / quick / full。

opts.model = struct( ...
    'simplify_pls_pcr', false, ...
    'pls_max_lv_cap', 40, ...
    'pcr_max_pc_cap', 40, ...
    'l1_l2_regularization', false, ...
    'l2_lambda', 1e-4, ...
    'dropout', false, ...
    'dropout_rate', 0.30, ...
    'early_stopping', false, ...
    'validation_patience', 8, ...
    ...
    'xgboost_simplify', false, ...
    'xgb_max_depth_cap', 4, ...
    'xgb_learn_rate_cap', 0.03, ...
    'xgb_min_leaf_floor', 8, ...
    'xgboost_grid_mode', 'quick', ...
    'xgb_num_learning_cycles', [], ...
    'xgb_learn_rates', [], ...
    'xgb_max_num_splits', [], ...
    'xgb_min_leaf_sizes', [], ...
    'cnn_simplify', false, ...
    'cnn_max_blocks', 2, ...
    'cnn_fc_units_cap', 48, ...
    'cnn_grid_mode', 'quick', ...
    'cnn_conv_channel_sets', {{}}, ...
    'cnn_fc_units_grid', [], ...
    'cnn_dropout_rates', [], ...
    'cnn_max_epochs_grid', [], ...
    'cnn_mini_batch_sizes', [], ...
    'cnn_initial_learn_rates', []);

% =========================
% 评估层配置 evaluation
% =========================
% enforce_small_rc_rp_gap
%   是否强调训练集与测试集表现差距不能过大。
%
% rc_rp_gap_threshold
%   R2_C 与 R2_P 的允许差值阈值。
opts.evaluation = struct( ...
    'enforce_small_rc_rp_gap', false, ...
    'rc_rp_gap_threshold', 0.05, ...
    'save_smooth_and_feature_plots', true);
end
