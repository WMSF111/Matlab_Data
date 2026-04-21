function model_result = train_model_with_generalization(dataset_mat_path, method_name, method_param, generalization_options)
% Train models with optional generalization improvements.

% 统一训练入口：在这里完成预处理、数据划分、特征选择、模型训练和结果落盘。
if nargin < 1 || isempty(dataset_mat_path)
    error('dataset_mat_path is required.');
end
if nargin < 2 || isempty(method_name)
    method_name = 'pls';
end
if nargin < 3
    method_param = [];
end
if nargin < 4 || isempty(generalization_options)
    generalization_options = default_generalization_options_local();
else
    generalization_options = normalize_generalization_options_local(generalization_options);
end

S = load(dataset_mat_path, 'dataset');
if ~isfield(S, 'dataset')
    error('dataset variable not found in %s', dataset_mat_path);
end
dataset = S.dataset;
X_raw_for_plot = dataset.X;
X = dataset.X;
y = double(dataset.y(:));
if size(X, 1) ~= numel(y)
    error('X and y size mismatch.');
end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(project_root));

method = lower(strtrim(method_name));
display_name = local_get_model_display_name(method);
dataset_tag = dataset.metadata.dataset_tag;
safe_dataset_tag = strrep(dataset_tag, '\', '_');
run_tag = getenv('HXR_RUN_TAG');
if isempty(run_tag)
    result_dir = fullfile(project_root, 'Result', 'Model', upper(method), safe_dataset_tag);
else
    result_dir = fullfile(project_root, 'Result', 'Model', ['Run_' run_tag], upper(method), safe_dataset_tag);
end
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% 先按数据元信息执行统一预处理，确保各模型使用同一份输入。
[X_all_pre, ~, dataset.metadata] = local_preprocess_split(X, X, dataset.metadata);
X = X_all_pre;

% 再按验证策略划分训练集和测试集；可选保留独立外部验证集。
[Xtrain, Ytrain, Xtest, Ytest, split_info] = local_make_train_test_split(X, y, generalization_options.validation);
used_external_validation = split_info.used_external_validation;

% 数据增强只作用于训练集，避免测试集信息泄漏。
if generalization_options.data_processing.data_augmentation
    [Xtrain, Ytrain] = local_augment_spectra(Xtrain, Ytrain, generalization_options.data_processing);
end

% 当当前数据组要求做特征选择时，特征选择必须严格限定在训练集内部。
if isfield(dataset.metadata, 'data_stage') && strcmpi(dataset.metadata.data_stage, 'selected')
    [Xtrain, Xtest, dataset.metadata, fs_note, Xtrain_stage1, Xtest_stage1] = local_apply_feature_pipeline(Xtrain, Ytrain, Xtest, dataset.metadata, generalization_options.feature, method);
else
    fs_note = 'no_feature_selection';
    Xtrain_stage1 = Xtrain;
    Xtest_stage1 = Xtest;
end
dataset.metadata.full_preprocessed_mean_curve = mean(X_all_pre, 1, 'omitnan');
dataset.metadata.full_preprocessed_feature_count = size(X_all_pre, 2);

stage1_model_result = local_train_model_dispatch(method, Xtrain_stage1, Ytrain, Xtest_stage1, Ytest, method_param, generalization_options.model, split_info.train_indices);
stage1_model_result.training_stage = 'stage1_feature_selection';
stage1_model_result.stage_description = '第一次降维：特征筛选后的模型';

model_result = local_train_model_dispatch(method, Xtrain, Ytrain, Xtest, Ytest, method_param, generalization_options.model, split_info.train_indices);
model_result.training_stage = 'stage2_feature_projection';
model_result.stage_description = '第二次降维：筛选后再降维/模型内部降维后的模型';
dataset.metadata = local_add_second_stage_band_markers(dataset.metadata, method, Xtrain, Ytrain, model_result, generalization_options.feature);

% 将本次训练结果按“特征算法_模型算法_时间_R2P百分值”保存。
time_tag = local_now_file_time_tag();
file_base = local_make_output_file_base(dataset.metadata, method, time_tag);
r2p_tag = local_r2p_to_percent_tag(model_result.R2_P);
result_stub = sprintf('%s_R2P%s', file_base, r2p_tag);
result_mat = fullfile(result_dir, [result_stub '.mat']);
plot_path = fullfile(result_dir, [result_stub '.tif']);
stage1_stub = sprintf('%s_stage1_R2P%s', file_base, local_r2p_to_percent_tag(stage1_model_result.R2_P));
stage2_stub = sprintf('%s_stage2_R2P%s', file_base, local_r2p_to_percent_tag(model_result.R2_P));
stage1_result_mat = fullfile(result_dir, [stage1_stub '.mat']);
stage2_result_mat = fullfile(result_dir, [stage2_stub '.mat']);
stage1_plot_path = fullfile(result_dir, [stage1_stub '.tif']);
stage2_plot_path = fullfile(result_dir, [stage2_stub '.tif']);
local_save_regression_plot(plot_path, display_name, model_result);
local_save_regression_plot(stage1_plot_path, [display_name ' Stage1'], stage1_model_result);
local_save_regression_plot(stage2_plot_path, [display_name ' Stage2'], model_result);
group_plot_dir = local_make_group_plot_dir(project_root, dataset.metadata, method, file_base);
save_smooth_feature_plots = local_should_save_smooth_feature_plots(generalization_options);
if save_smooth_feature_plots && ~exist(group_plot_dir, 'dir')
    mkdir(group_plot_dir);
end
smooth_plot_path = local_save_smooth_overview(group_plot_dir, file_base, X_raw_for_plot, X_all_pre, dataset.metadata, save_smooth_feature_plots);
feature_plot_path = local_save_selected_band_overlay(group_plot_dir, file_base, X_all_pre, dataset.metadata, save_smooth_feature_plots);

stage1_model_result.dataset_mat_path = dataset_mat_path;
stage1_model_result.dataset_tag = dataset_tag;
stage1_model_result.dataset_metadata = dataset.metadata;
stage1_model_result.method_name = method;
stage1_model_result.model_display_name = display_name;
stage1_model_result.result_mat_path = stage1_result_mat;
stage1_model_result.regression_plot_path = stage1_plot_path;
stage1_model_result.used_external_validation = used_external_validation;
stage1_model_result.used_nested_cv_selection = generalization_options.feature.nested_cv_selection;
stage1_model_result.used_stable_feature_selection = generalization_options.feature.use_stable_features;
stage1_model_result.used_data_augmentation = generalization_options.data_processing.data_augmentation;
stage1_model_result.generalization_note = [fs_note '_stage1'];
stage1_model_result.generalization_options = generalization_options;

model_result.dataset_mat_path = dataset_mat_path;
model_result.dataset_tag = dataset_tag;
model_result.dataset_metadata = dataset.metadata;
model_result.method_name = method;
model_result.model_display_name = display_name;
model_result.result_mat_path = result_mat;
model_result.regression_plot_path = plot_path;
model_result.smooth_plot_path = smooth_plot_path;
model_result.selected_band_plot_path = feature_plot_path;
model_result.used_external_validation = used_external_validation;
model_result.used_nested_cv_selection = generalization_options.feature.nested_cv_selection;
model_result.used_stable_feature_selection = generalization_options.feature.use_stable_features;
model_result.used_data_augmentation = generalization_options.data_processing.data_augmentation;
model_result.generalization_note = fs_note;
model_result.generalization_options = generalization_options;
model_result.stage1_result_mat_path = stage1_result_mat;
model_result.stage2_result_mat_path = stage2_result_mat;
model_result.stage1_regression_plot_path = stage1_plot_path;
model_result.stage2_regression_plot_path = stage2_plot_path;
model_result.stage1_R2_C = stage1_model_result.R2_C;
model_result.stage1_R2_P = stage1_model_result.R2_P;
model_result.stage1_RMSEC = stage1_model_result.RMSEC;
model_result.stage1_RMSEP = stage1_model_result.RMSEP;
model_result.stage1_RPD = stage1_model_result.RPD;
model_result.stage2_R2_C = model_result.R2_C;
model_result.stage2_R2_P = model_result.R2_P;
model_result.stage2_RMSEC = model_result.RMSEC;
model_result.stage2_RMSEP = model_result.RMSEP;
model_result.stage2_RPD = model_result.RPD;

stage2_model_result = model_result; %#ok<NASGU>
save(stage1_result_mat, 'stage1_model_result');
save(stage2_result_mat, 'stage2_model_result');
save(result_mat, 'model_result', 'stage1_model_result', 'stage2_model_result');
end

function [Xtrain, Ytrain, Xtest, Ytest, info] = local_make_train_test_split(X, y, validation_opts)
% 按配置划分训练/测试：
% 开启外部验证时使用随机留出，否则沿用 KS 划分。
n = size(X, 1);
info = struct();
if validation_opts.use_external_holdout && n >= 10
    rng(validation_opts.random_seed);
    cvp = cvpartition(n, 'HoldOut', validation_opts.external_ratio);
    train_mask = training(cvp);
    test_mask = test(cvp);
    Xtrain = X(train_mask, :);
    Ytrain = y(train_mask, :);
    Xtest = X(test_mask, :);
    Ytest = y(test_mask, :);
    info.used_external_validation = true;
    info.train_indices = find(train_mask);
else
    ks_count = max(2, round(n * 0.75));
    [select, not_select] = KS(X, ks_count);
    Xtrain = X(select, :);
    Ytrain = y(select, :);
    Xtest = X(not_select, :);
    Ytest = y(not_select, :);
    info.used_external_validation = false;
    info.train_indices = select;
end
end

function [Xtrain_out, Xtest_out, metadata, note, Xtrain_stage1, Xtest_stage1] = local_apply_feature_pipeline(Xtrain, Ytrain, Xtest, metadata, feature_opts, model_method)
% 训练集内部特征处理：
% 可以选择普通筛选，也可以走嵌套交叉验证与稳定特征策略。
if nargin < 6
    model_method = '';
end
method = lower(strtrim(char(metadata.fs_method)));
param = metadata.fs_param;
note = 'train_only_feature_selection';
if feature_opts.nested_cv_selection || feature_opts.use_stable_features
    fs_result = local_nested_feature_selection(Xtrain, Ytrain, method, param, feature_opts);
    note = 'nested_cv_feature_selection';
else
    fs_result = select_features_by_method(Xtrain, Ytrain, method, param);
end

metadata.feature_score = fs_result.score;
metadata.feature_info = fs_result.info;
metadata.selection_scope = note;
if isfield(fs_result.info, 'selection_frequency')
    metadata.feature_info.selection_frequency = fs_result.info.selection_frequency;
end

selected_idx = fs_result.selected_idx(:)';
Xtrain_out = Xtrain(:, selected_idx);
Xtest_out = Xtest(:, selected_idx);
Xtrain_stage1 = Xtrain_out;
Xtest_stage1 = Xtest_out;
metadata.selected_idx = selected_idx;
metadata.pre_pca_feature_count = numel(selected_idx);
metadata.stage1_selected_idx = selected_idx;
if isfield(metadata, 'used_band_idx') && ~isempty(metadata.used_band_idx)
    base_band_idx = metadata.used_band_idx(:)';
    metadata.pre_feature_used_band_idx = base_band_idx;
    selected_idx = selected_idx(selected_idx >= 1 & selected_idx <= numel(base_band_idx));
    metadata.used_band_idx = base_band_idx(selected_idx);
    metadata.stage1_used_band_idx = metadata.used_band_idx;
    metadata.final_used_band_idx = metadata.used_band_idx;
    if ~isempty(metadata.used_band_idx)
        metadata.used_band_range = [min(metadata.used_band_idx), max(metadata.used_band_idx)];
    end
end
metadata.final_feature_count = size(Xtrain_out, 2);
metadata.post_feature_pca_projection = false;
metadata.post_feature_pca_skip_reason = '';

% PLS 和 PCR 本身会通过潜变量/主成分做降维，因此这里不再追加 PCA 投影。
skip_post_pca = any(strcmpi(model_method, {'pls', 'pcr'}));
if skip_post_pca
    metadata.post_feature_pca_skip_reason = 'PLS/PCR use internal dimensionality reduction';
elseif isfield(feature_opts, 'post_feature_pca_projection') && feature_opts.post_feature_pca_projection
    target_components = local_get_feature_option(feature_opts, 'post_feature_pca_components', 20);
    if size(Xtrain_out, 2) <= target_components
        metadata.post_feature_pca_skip_reason = sprintf('Stage-1 feature count %d <= PCA target %d', size(Xtrain_out, 2), target_components);
    else
        [Xtrain_out, Xtest_out, pca_info] = local_apply_post_feature_pca_projection(Xtrain_out, Xtest_out, feature_opts);
        metadata.post_feature_pca_projection = true;
        metadata.post_feature_pca_info = pca_info;
        metadata.final_feature_count = size(Xtrain_out, 2);
    end
end
end

function [Xtrain_pca, Xtest_pca, pca_info] = local_apply_post_feature_pca_projection(Xtrain, Xtest, feature_opts)
% 在特征筛选之后再做 PCA 主成分投影：
% 先用 CARS/SPA/corr_topk/PCA波段筛选保留一批候选特征，
% 再只用训练集拟合 PCA，并把训练集/测试集投影到同一个主成分空间。
max_comp = local_get_feature_option(feature_opts, 'post_feature_pca_components', 20);
max_comp = max(1, round(max_comp));
max_allowed = min([size(Xtrain, 2), max(size(Xtrain, 1) - 1, 1)]);
component_count = min(max_comp, max_allowed);

[coeff, score, latent, ~, explained, mu] = pca(Xtrain, 'NumComponents', component_count);
Xtrain_pca = score(:, 1:component_count);
Xtest_pca = (Xtest - mu) * coeff(:, 1:component_count);

pca_info = struct();
pca_info.mode = 'post_feature_pca_projection';
pca_info.input_feature_count = size(Xtrain, 2);
pca_info.component_count = component_count;
pca_info.explained = explained(:)';
pca_info.cumulative_explained = cumsum(explained(:)');
pca_info.latent = latent(:)';
pca_info.mu = mu;
pca_info.coeff = coeff;
end

function metadata = local_add_second_stage_band_markers(metadata, method, Xtrain_final, Ytrain, model_result, feature_opts)
% 为右侧“第二阶段”特征图准备可映射回原始波段的高贡献点。
% PLS/PCR 不是真的再选原始波段，这里标的是内部降维载荷贡献最高的原始波段。
metadata.second_stage_used_band_idx = [];
metadata.second_stage_note = '';
if ~isfield(metadata, 'stage1_used_band_idx') || isempty(metadata.stage1_used_band_idx)
    return;
end
base_band_idx = metadata.stage1_used_band_idx(:)';
method = lower(strtrim(method));
switch method
    case 'pcr'
        component_count = max(1, round(model_result.A));
        contribution = local_pcr_feature_contribution(Xtrain_final, component_count);
        marker_count = min(numel(base_band_idx), max(component_count, local_second_stage_marker_count(feature_opts)));
        metadata.second_stage_used_band_idx = local_top_contribution_bands(base_band_idx, contribution, marker_count);
        metadata.second_stage_note = sprintf('PCR top loading bands, PC=%d', component_count);
    case 'pls'
        component_count = max(1, round(model_result.A));
        contribution = local_pls_feature_contribution(Xtrain_final, Ytrain, component_count);
        marker_count = min(numel(base_band_idx), max(component_count, local_second_stage_marker_count(feature_opts)));
        metadata.second_stage_used_band_idx = local_top_contribution_bands(base_band_idx, contribution, marker_count);
        metadata.second_stage_note = sprintf('PLS top loading bands, LV=%d', component_count);
    otherwise
        if isfield(metadata, 'post_feature_pca_projection') && metadata.post_feature_pca_projection && ...
                isfield(metadata, 'post_feature_pca_info') && isfield(metadata.post_feature_pca_info, 'coeff')
            coeff = metadata.post_feature_pca_info.coeff;
            component_count = metadata.post_feature_pca_info.component_count;
            contribution = sum(abs(coeff(:, 1:component_count)), 2);
            marker_count = min(numel(base_band_idx), max(component_count, local_second_stage_marker_count(feature_opts)));
            metadata.second_stage_used_band_idx = local_top_contribution_bands(base_band_idx, contribution, marker_count);
            metadata.second_stage_note = sprintf('Post PCA top loading bands, PC=%d', component_count);
        else
            metadata.second_stage_used_band_idx = base_band_idx;
            metadata.second_stage_note = 'No second-stage projection; same as stage-1 selected bands';
        end
end
end

function marker_count = local_second_stage_marker_count(feature_opts)
marker_count = 20;
if isfield(feature_opts, 'post_feature_pca_components') && ~isempty(feature_opts.post_feature_pca_components)
    marker_count = max(1, round(feature_opts.post_feature_pca_components));
end
end

function contribution = local_pcr_feature_contribution(Xtrain, component_count)
component_count = min([component_count, size(Xtrain, 2), max(size(Xtrain, 1) - 1, 1)]);
[coeff, ~] = pca(Xtrain, 'NumComponents', component_count);
contribution = sum(abs(coeff(:, 1:component_count)), 2);
end

function contribution = local_pls_feature_contribution(Xtrain, Ytrain, component_count)
component_count = min([component_count, size(Xtrain, 2), max(size(Xtrain, 1) - 1, 1)]);
try
    [XL, ~] = plsregress(Xtrain, Ytrain(:), component_count);
    contribution = sum(abs(XL(:, 1:component_count)), 2);
catch
    contribution = local_pcr_feature_contribution(Xtrain, component_count);
end
end

function band_idx = local_top_contribution_bands(base_band_idx, contribution, marker_count)
contribution = contribution(:)';
valid_count = min(numel(base_band_idx), numel(contribution));
base_band_idx = base_band_idx(1:valid_count);
contribution = contribution(1:valid_count);
contribution(~isfinite(contribution)) = 0;
[~, order] = sort(contribution, 'descend');
marker_count = min(marker_count, numel(order));
band_idx = unique(base_band_idx(order(1:marker_count)), 'stable');
end

function value = local_get_feature_option(feature_opts, field_name, default_value)
if isfield(feature_opts, field_name) && ~isempty(feature_opts.(field_name))
    value = feature_opts.(field_name);
else
    value = default_value;
end
end

function fs_result = local_nested_feature_selection(X, y, method, param, feature_opts)
% 嵌套式特征筛选：
% 在内层折上重复选特征，再按出现频率汇总稳定特征。
n = size(X, 1);
nfold = max(2, min(feature_opts.inner_cv_folds, n));
fold_id = local_make_fold_id(n, nfold, 11);
n_features = size(X, 2);
count = zeros(1, n_features);
base_score = zeros(1, n_features);
selected_size = zeros(nfold, 1);
for f = 1:nfold
    tr = fold_id ~= f;
    fs_one = select_features_by_method(X(tr, :), y(tr), method, param);
    idx = unique(fs_one.selected_idx(:)', 'stable');
    idx = idx(idx >= 1 & idx <= n_features);
    count(idx) = count(idx) + 1;
    selected_size(f) = numel(idx);
    if numel(fs_one.score) == n_features
        base_score = base_score + reshape(fs_one.score, 1, []);
    else
        base_score(idx) = base_score(idx) + 1;
    end
end
freq = count / nfold;
target_k = local_target_feature_count(method, param, n_features, selected_size);
if feature_opts.use_stable_features
    selected_idx = find(freq >= feature_opts.stability_threshold);
    if numel(selected_idx) > target_k
        [~, order] = sort(freq(selected_idx), 'descend');
        selected_idx = selected_idx(order(1:target_k));
    end
else
    [~, order] = sortrows([-freq(:), -base_score(:)]);
    selected_idx = order(1:min(target_k, numel(order)))';
end
if isempty(selected_idx)
    [~, order] = sort(freq, 'descend');
    selected_idx = order(1:min(target_k, numel(order)))';
end
fs_result = struct();
fs_result.method = method;
fs_result.selected_idx = unique(selected_idx(:)', 'stable');
fs_result.score = base_score / nfold;
fs_result.info = struct('selection_frequency', freq, 'target_k', target_k, 'inner_cv_folds', nfold);
end

function target_k = local_target_feature_count(method, param, n_features, selected_size)
target_k = max(1, round(median(selected_size(selected_size > 0))));
if isempty(target_k) || ~isfinite(target_k)
    target_k = min(10, n_features);
end
switch lower(method)
    case {'corr_topk', 'pca', 'spa'}
        if isnumeric(param) && ~isempty(param)
            target_k = min(n_features, max(1, round(param)));
        end
    case 'cars'
        if isstruct(param) && isfield(param, 'target_count') && ~isempty(param.target_count)
            target_k = min(n_features, max(1, round(param.target_count)));
        end
end
end

function [Xaug, Yaug] = local_augment_spectra(X, y, dp_opts)
% 光谱增强：通过小幅平移与高斯噪声扩充训练样本。
Xaug = X;
Yaug = y(:);
for c = 1:max(1, dp_opts.augmentation_copies)
    shift = randi([-dp_opts.max_shift, dp_opts.max_shift], size(X, 1), 1);
    Xshift = zeros(size(X));
    for i = 1:size(X, 1)
        Xshift(i, :) = circshift(X(i, :), shift(i), 2);
    end
    noise_scale = dp_opts.noise_std * max(std(X, 0, 2), eps);
    noise = randn(size(X)) .* noise_scale;
    Xaug = [Xaug; Xshift + noise]; %#ok<AGROW>
    Yaug = [Yaug; y(:)]; %#ok<AGROW>
end
end

function [Ypred_train, Ypred, best_detail, CV, best_A] = local_train_pls(Xtrain, Ytrain, Xtest, Ytest, method_param, model_opts)
% PLS 训练：用训练集内部交叉验证选择潜变量数。
max_lv = min(256, size(Xtrain, 2));
cv_fold = 10;
if isstruct(method_param)
    if isfield(method_param, 'max_lv') && ~isempty(method_param.max_lv)
        max_lv = min(max_lv, max(1, round(method_param.max_lv)));
    end
    if isfield(method_param, 'cv_fold') && ~isempty(method_param.cv_fold)
        cv_fold = max(2, round(method_param.cv_fold));
    end
end
if model_opts.simplify_pls_pcr
    max_lv = min(max_lv, model_opts.pls_max_lv_cap);
    cv_fold = min(cv_fold, 5);
end
CV = plscv(Xtrain, Ytrain, max_lv, cv_fold, 'center');
best_A = CV.optLV;
PLS = pls(Xtrain, Ytrain, best_A, 'center');
Ypred = plsval(PLS, Xtest, Ytest);
Ypred_train = plsval(PLS, Xtrain, Ytrain);
best_detail = sprintf('PLS A=%d, maxLV=%d, cv=%d', best_A, max_lv, cv_fold);
end

function model_result = local_train_model_dispatch(method, Xtrain, Ytrain, Xtest, Ytest, method_param, model_opts, train_indices)
% 根据模型名称分发训练，用于第一阶段和第二阶段复用同一套训练逻辑。
switch lower(strtrim(method))
    case 'pls'
        [Ypred_train, Ypred, best_detail, cv_info, latent_value] = local_train_pls(Xtrain, Ytrain, Xtest, Ytest, method_param, model_opts);
        model_result = build_result_struct(method, latent_value, [], train_indices, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, cv_info);
        model_result.best_param_detail = best_detail;
    case 'pcr'
        [Ypred_train, Ypred, best_detail, latent_value] = local_train_pcr(Xtrain, Ytrain, Xtest, Ytest, method_param, model_opts);
        model_result = build_result_struct(method, latent_value, 1:latent_value, train_indices, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = best_detail;
    case 'svr'
        [Ypred_train, Ypred, best_detail, best_cfg] = local_train_svr(Xtrain, Ytrain, Xtest, method_param);
        model_result = build_result_struct(method, NaN, best_cfg, train_indices, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = best_detail;
    case 'rf'
        [Ypred_train, Ypred, best_detail, best_cfg] = local_train_rf(Xtrain, Ytrain, Xtest, method_param);
        model_result = build_result_struct(method, NaN, best_cfg, train_indices, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = best_detail;
    case 'gpr'
        [Ypred_train, Ypred, best_detail, best_cfg] = local_train_gpr(Xtrain, Ytrain, Xtest, method_param);
        model_result = build_result_struct(method, NaN, best_cfg, train_indices, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = best_detail;
    case 'knn'
        [Ypred_train, Ypred, best_detail, best_cfg] = local_train_knn(Xtrain, Ytrain, Xtest, method_param);
        model_result = build_result_struct(method, NaN, best_cfg, train_indices, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = best_detail;
    case 'xgboost'
        [Ypred_train, Ypred, best_detail, best_cfg] = local_train_xgboost(Xtrain, Ytrain, Xtest, method_param, model_opts);
        model_result = build_result_struct(method, NaN, best_cfg, train_indices, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = best_detail;
    case 'cnn'
        [Ypred_train, Ypred, best_detail, cnn_info] = local_train_cnn(Xtrain, Ytrain, Xtest, Ytest, method_param, model_opts);
        model_result = build_result_struct(method, NaN, cnn_info, train_indices, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = best_detail;
    otherwise
        error('Unsupported method_name: %s', method);
end
end

function [Ypred_train, Ypred, best_detail, best_k] = local_train_pcr(Xtrain, Ytrain, Xtest, ~, method_param, model_opts)
% PCR 训练：先 PCA，再在训练集内部搜索最优主成分个数。
max_pc = min([200, size(Xtrain, 2), size(Xtrain, 1) - 1]);
if isnumeric(method_param) && ~isempty(method_param)
    max_pc = min(max_pc, max(1, round(method_param)));
elseif isstruct(method_param) && isfield(method_param, 'max_pc') && ~isempty(method_param.max_pc)
    max_pc = min(max_pc, max(1, round(method_param.max_pc)));
end
if model_opts.simplify_pls_pcr
    max_pc = min(max_pc, model_opts.pcr_max_pc_cap);
end
[coeff, score, ~, ~, ~, mu] = pca(Xtrain);
best_cv_rmse = inf;
best_k = 1;
best_beta = [];
for k = 1:max_pc
    score_train = score(:, 1:k);
    cv_rmse = local_pcr_cv_rmse(score_train, Ytrain, 5);
    if cv_rmse < best_cv_rmse
        best_cv_rmse = cv_rmse;
        best_k = k;
        best_beta = [ones(size(score_train, 1), 1), score_train] \ Ytrain;
    end
end
score_train = score(:, 1:best_k);
score_test = bsxfun(@minus, Xtest, mu) * coeff(:, 1:best_k);
Ypred_train = [ones(size(score_train, 1), 1), score_train] * best_beta;
Ypred = [ones(size(score_test, 1), 1), score_test] * best_beta;
best_detail = sprintf('PCR k=%d, trainCV=%.4f', best_k, best_cv_rmse);
end

function [Ypred_train, Ypred, best_detail, best_cfg] = local_train_svr(Xtrain, Ytrain, Xtest, method_param)
% SVR 训练：在训练集内部搜索核函数、惩罚系数和尺度。
grid = build_svr_grid(method_param);
best_cv_rmse = inf;
best_cfg = grid(1);
for i = 1:numel(grid)
    cfg = grid(i);
    cv_rmse = local_svr_cv_rmse(Xtrain, Ytrain, cfg, 5);
    if cv_rmse < best_cv_rmse
        best_cv_rmse = cv_rmse;
        best_cfg = cfg;
    end
end
mdl = fitrsvm(Xtrain, Ytrain, ...
    'KernelFunction', best_cfg.kernel, ...
    'BoxConstraint', best_cfg.box, ...
    'KernelScale', best_cfg.scale, ...
    'Standardize', true);
Ypred_train = predict(mdl, Xtrain);
Ypred = predict(mdl, Xtest);
best_detail = sprintf('SVR kernel=%s, box=%g, scale=%s, trainCV=%.4f', best_cfg.kernel, best_cfg.box, scale_to_text(best_cfg.scale), best_cv_rmse);
end

function [Ypred_train, Ypred, best_detail, best_cfg] = local_train_rf(Xtrain, Ytrain, Xtest, method_param)
% RF 训练：搜索树数量与最小叶节点大小。
grid = build_rf_grid(method_param);
best_cv_rmse = inf;
best_cfg = grid(1);
for i = 1:numel(grid)
    cfg = grid(i);
    cv_rmse = local_rf_cv_rmse(Xtrain, Ytrain, cfg, 5);
    if cv_rmse < best_cv_rmse
        best_cv_rmse = cv_rmse;
        best_cfg = cfg;
    end
end
mdl = TreeBagger(best_cfg.num_trees, Xtrain, Ytrain, ...
    'Method', 'regression', ...
    'MinLeafSize', best_cfg.min_leaf, ...
    'OOBPrediction', 'off');
Ypred_train = predict(mdl, Xtrain);
Ypred = predict(mdl, Xtest);
if iscell(Ypred_train), Ypred_train = str2double(Ypred_train); end
if iscell(Ypred), Ypred = str2double(Ypred); end
Ypred_train = Ypred_train(:);
Ypred = Ypred(:);
best_detail = sprintf('RF trees=%d, minLeaf=%d, trainCV=%.4f', best_cfg.num_trees, best_cfg.min_leaf, best_cv_rmse);
end

function [Ypred_train, Ypred, best_detail, best_cfg] = local_train_gpr(Xtrain, Ytrain, Xtest, method_param)
% GPR 训练：搜索不同核函数。
grid = build_gpr_grid(method_param);
best_cv_rmse = inf;
best_cfg = grid(1);
for i = 1:numel(grid)
    cfg = grid(i);
    cv_rmse = local_gpr_cv_rmse(Xtrain, Ytrain, cfg, 5);
    if cv_rmse < best_cv_rmse
        best_cv_rmse = cv_rmse;
        best_cfg = cfg;
    end
end
mdl = fitrgp(Xtrain, Ytrain, ...
    'KernelFunction', best_cfg.kernel, ...
    'Standardize', true);
Ypred_train = predict(mdl, Xtrain);
Ypred = predict(mdl, Xtest);
best_detail = sprintf('GPR kernel=%s, trainCV=%.4f', best_cfg.kernel, best_cv_rmse);
end

function [Ypred_train, Ypred, best_detail, best_cfg] = local_train_knn(Xtrain, Ytrain, Xtest, method_param)
% KNN 训练：搜索邻居数、距离度量和加权方式。
grid = build_knn_grid(method_param);
best_cv_rmse = inf;
best_cfg = grid(1);
for i = 1:numel(grid)
    cfg = grid(i);
    cv_rmse = local_knn_cv_rmse(Xtrain, Ytrain, cfg, 5);
    if cv_rmse < best_cv_rmse
        best_cv_rmse = cv_rmse;
        best_cfg = cfg;
    end
end
Ypred_train = local_knn_predict(Xtrain, Ytrain, Xtrain, best_cfg.k, best_cfg.distance, best_cfg.weighting, true);
Ypred = local_knn_predict(Xtrain, Ytrain, Xtest, best_cfg.k, best_cfg.distance, best_cfg.weighting, false);
best_detail = sprintf('KNN k=%d, distance=%s, weighting=%s, trainCV=%.4f', best_cfg.k, best_cfg.distance, best_cfg.weighting, best_cv_rmse);
end

function [Ypred_train, Ypred, best_detail, best_cfg] = local_train_xgboost(Xtrain, Ytrain, Xtest, method_param, model_opts)
% XGBoost 风格回归：
% 这里使用 MATLAB 的 LSBoost 回归树来近似实现梯度提升树。
grid = build_xgboost_grid(method_param, model_opts);
best_cv_rmse = inf;
best_cfg = grid(1);
for i = 1:numel(grid)
    cfg = grid(i);
    cv_rmse = local_xgboost_cv_rmse(Xtrain, Ytrain, cfg, 5);
    if cv_rmse < best_cv_rmse
        best_cv_rmse = cv_rmse;
        best_cfg = cfg;
    end
end
tree = templateTree('MaxNumSplits', best_cfg.max_num_splits, 'MinLeafSize', best_cfg.min_leaf_size);
mdl = fitrensemble(Xtrain, Ytrain, ...
    'Method', 'LSBoost', ...
    'Learners', tree, ...
    'NumLearningCycles', best_cfg.num_learning_cycles, ...
    'LearnRate', best_cfg.learn_rate);
Ypred_train = predict(mdl, Xtrain);
Ypred = predict(mdl, Xtest);
best_detail = sprintf('XGBoost-like LSBoost cycles=%d, lr=%.4f, maxSplits=%d, minLeaf=%d, trainCV=%.4f', ...
    best_cfg.num_learning_cycles, best_cfg.learn_rate, best_cfg.max_num_splits, best_cfg.min_leaf_size, best_cv_rmse);
end

function [Ypred_train, Ypred, best_detail, cnn_info] = local_train_cnn(Xtrain, Ytrain, Xtest, Ytest, method_param, model_opts)
% 1D CNN 回归：
% 根据开关动态调整卷积块数量、Dropout、L2 和早停策略。
cfg = normalize_cnn_param(method_param);
if model_opts.cnn_simplify
    cfg.conv_channels = cfg.conv_channels(1:min(numel(cfg.conv_channels), model_opts.cnn_max_blocks));
    cfg.fc_units = min(cfg.fc_units, model_opts.cnn_fc_units_cap);
    cfg.max_epochs = min(cfg.max_epochs, 100);
end
if model_opts.dropout
    cfg.dropout_rate = model_opts.dropout_rate;
else
    cfg.dropout_rate = 0;
end

[XsubTrain, YsubTrain, Xval, Yval] = local_make_cnn_validation_split(Xtrain, Ytrain);
layers = local_build_cnn_layers(size(Xtrain, 2), cfg, model_opts);
options = local_build_cnn_training_options(cfg, model_opts, Xval, Yval);

net = trainNetwork(local_matrix_to_sequence_cells(XsubTrain), YsubTrain, layers, options);
Ypred_train = predict(net, local_matrix_to_sequence_cells(Xtrain), 'MiniBatchSize', cfg.mini_batch_size);
Ypred = predict(net, local_matrix_to_sequence_cells(Xtest), 'MiniBatchSize', cfg.mini_batch_size);
Ypred_train = double(Ypred_train(:));
Ypred = double(Ypred(:));

best_detail = sprintf('CNN conv=%s, fc=%d, dropout=%.2f, epochs=%d, lr=%g%s%s', ...
    mat2str(cfg.conv_channels), cfg.fc_units, cfg.dropout_rate, cfg.max_epochs, cfg.initial_learn_rate, ...
    ternary_suffix(model_opts.l1_l2_regularization, sprintf(', L2=%g', model_opts.l2_lambda)), ...
    ternary_suffix(model_opts.early_stopping, sprintf(', earlyStop=%d', model_opts.validation_patience)));
cnn_info = struct('conv_channels', cfg.conv_channels, 'fc_units', cfg.fc_units, 'dropout_rate', cfg.dropout_rate, 'validation_count', numel(Yval));
if nnz(isfinite(Ytest)) < 2
    warning('CNN test labels have fewer than 2 finite values.');
end
end

function grid = build_xgboost_grid(method_param, model_opts)
% 构造 XGBoost 风格参数网格，并根据简化开关收缩搜索范围。
if nargin < 1 || isempty(method_param)
    method_param = struct();
end
cycles = [300 500];
rates = [0.05 0.03];
splits = [15 31];
leaves = [3 5];
if isstruct(method_param)
    if isfield(method_param, 'num_learning_cycles') && ~isempty(method_param.num_learning_cycles)
        cycles = unique(max(50, round(method_param.num_learning_cycles(:)')));
    end
    if isfield(method_param, 'learn_rate') && ~isempty(method_param.learn_rate)
        rates = unique(max(1e-3, method_param.learn_rate(:)'));
    end
    if isfield(method_param, 'max_num_splits') && ~isempty(method_param.max_num_splits)
        splits = unique(max(1, round(method_param.max_num_splits(:)')));
    end
    if isfield(method_param, 'min_leaf_size') && ~isempty(method_param.min_leaf_size)
        leaves = unique(max(1, round(method_param.min_leaf_size(:)')));
    end
end
if model_opts.xgboost_simplify
    rates = min(rates, model_opts.xgb_learn_rate_cap);
    splits = min(splits, 2^model_opts.xgb_max_depth_cap - 1);
    leaves = max(leaves, model_opts.xgb_min_leaf_floor);
    cycles = min(cycles, 250);
end
idx = 0;
for i = 1:numel(cycles)
    for j = 1:numel(rates)
        for k = 1:numel(splits)
            for t = 1:numel(leaves)
                idx = idx + 1;
                grid(idx).num_learning_cycles = cycles(i); %#ok<AGROW>
                grid(idx).learn_rate = rates(j); %#ok<AGROW>
                grid(idx).max_num_splits = splits(k); %#ok<AGROW>
                grid(idx).min_leaf_size = leaves(t); %#ok<AGROW>
            end
        end
    end
end
end

function cfg = normalize_cnn_param(method_param)
% 归一化 CNN 参数，补齐默认值并限制到合理范围。
cfg = struct('conv_channels', [16 32], 'fc_units', 64, 'dropout_rate', 0.20, 'max_epochs', 120, 'mini_batch_size', 16, 'initial_learn_rate', 1e-3);
if isstruct(method_param)
    names = fieldnames(method_param);
    for i = 1:numel(names)
        cfg.(names{i}) = method_param.(names{i});
    end
end
cfg.conv_channels = reshape(cfg.conv_channels, 1, []);
cfg.fc_units = max(8, round(cfg.fc_units));
cfg.max_epochs = max(10, round(cfg.max_epochs));
cfg.mini_batch_size = max(4, round(cfg.mini_batch_size));
cfg.dropout_rate = min(max(cfg.dropout_rate, 0), 0.8);
end

function grid = build_svr_grid(method_param)
kernels = {'linear', 'gaussian'};
boxes = [1, 10];
scales = {'auto'};
if isnumeric(method_param) && ~isempty(method_param)
    boxes = unique(max(0.1, method_param(:)'));
elseif ischar(method_param) && strcmpi(method_param, 'fast')
    boxes = [1, 10];
elseif isstruct(method_param)
    if isfield(method_param, 'kernels') && ~isempty(method_param.kernels)
        kernels = method_param.kernels;
    end
    if isfield(method_param, 'boxes') && ~isempty(method_param.boxes)
        boxes = unique(max(0.1, method_param.boxes(:)'));
    end
    if isfield(method_param, 'scales') && ~isempty(method_param.scales)
        scales = method_param.scales;
    end
end
idx = 0;
for i = 1:numel(kernels)
    for j = 1:numel(boxes)
        for k = 1:numel(scales)
            idx = idx + 1;
            grid(idx).kernel = kernels{i}; %#ok<AGROW>
            grid(idx).box = boxes(j); %#ok<AGROW>
            grid(idx).scale = scales{k}; %#ok<AGROW>
        end
    end
end
end

function grid = build_rf_grid(method_param)
num_trees_list = [100, 200];
min_leaf_list = [1, 5, 10];
if isnumeric(method_param) && ~isempty(method_param)
    num_trees_list = unique(max(10, round(method_param(:)')));
elseif isstruct(method_param)
    if isfield(method_param, 'num_trees') && ~isempty(method_param.num_trees)
        num_trees_list = unique(max(10, round(method_param.num_trees(:)')));
    end
    if isfield(method_param, 'min_leaf') && ~isempty(method_param.min_leaf)
        min_leaf_list = unique(max(1, round(method_param.min_leaf(:)')));
    end
end
idx = 0;
for i = 1:numel(num_trees_list)
    for j = 1:numel(min_leaf_list)
        idx = idx + 1;
        grid(idx).num_trees = num_trees_list(i); %#ok<AGROW>
        grid(idx).min_leaf = min_leaf_list(j); %#ok<AGROW>
    end
end
end

function grid = build_gpr_grid(method_param)
kernels = {'squaredexponential', 'ardsquaredexponential'};
if ischar(method_param) && strcmpi(method_param, 'fast')
    kernels = {'squaredexponential'};
elseif iscell(method_param) && ~isempty(method_param)
    kernels = method_param;
elseif isstruct(method_param)
    if isfield(method_param, 'kernels') && ~isempty(method_param.kernels)
        kernels = method_param.kernels;
    end
end
for i = 1:numel(kernels)
    grid(i).kernel = kernels{i}; %#ok<AGROW>
end
end

function grid = build_knn_grid(method_param)
ks = [3, 5, 7, 9];
distances = {'euclidean', 'cityblock'};
weightings = {'uniform', 'inverse'};
if isnumeric(method_param) && ~isempty(method_param)
    ks = unique(max(1, round(method_param(:)')));
elseif isstruct(method_param)
    if isfield(method_param, 'k') && ~isempty(method_param.k)
        ks = unique(max(1, round(method_param.k(:)')));
    end
    if isfield(method_param, 'distances') && ~isempty(method_param.distances)
        distances = method_param.distances;
    end
    if isfield(method_param, 'weightings') && ~isempty(method_param.weightings)
        weightings = method_param.weightings;
    end
end
idx = 0;
for i = 1:numel(ks)
    for j = 1:numel(distances)
        for k = 1:numel(weightings)
            idx = idx + 1;
            grid(idx).k = ks(i); %#ok<AGROW>
            grid(idx).distance = distances{j}; %#ok<AGROW>
            grid(idx).weighting = weightings{k}; %#ok<AGROW>
        end
    end
end
end

function txt = scale_to_text(scale)
if isnumeric(scale)
    txt = num2str(scale);
else
    txt = char(string(scale));
end
end

function [XsubTrain, YsubTrain, Xval, Yval] = local_make_cnn_validation_split(Xtrain, Ytrain)
n = size(Xtrain, 1);
if n < 8
    XsubTrain = Xtrain;
    YsubTrain = Ytrain;
    Xval = Xtrain;
    Yval = Ytrain;
    return;
end
cvp = cvpartition(n, 'HoldOut', 0.15);
train_mask = training(cvp);
val_mask = test(cvp);
XsubTrain = Xtrain(train_mask, :);
YsubTrain = Ytrain(train_mask);
Xval = Xtrain(val_mask, :);
Yval = Ytrain(val_mask);
end

function layers = local_build_cnn_layers(feature_count, cfg, model_opts)
% 按当前配置动态搭建 1D CNN 回归网络结构。
layers = [sequenceInputLayer(1, 'Name', 'input', 'MinLength', feature_count)];
for i = 1:numel(cfg.conv_channels)
    block = [
        convolution1dLayer(3, cfg.conv_channels(i), 'Padding', 'same', 'Name', sprintf('conv_%d', i))
        batchNormalizationLayer('Name', sprintf('bn_%d', i))
        reluLayer('Name', sprintf('relu_%d', i))];
    layers = [layers; block]; %#ok<AGROW>
    if model_opts.dropout && cfg.dropout_rate > 0
        layers = [layers; dropoutLayer(cfg.dropout_rate, 'Name', sprintf('drop_%d', i))]; %#ok<AGROW>
    end
end
tail = [
    globalAveragePooling1dLayer('Name', 'gap')
    fullyConnectedLayer(cfg.fc_units, 'Name', 'fc1')
    reluLayer('Name', 'fc_relu')];
layers = [layers; tail]; %#ok<AGROW>
if model_opts.dropout && cfg.dropout_rate > 0
    layers = [layers; dropoutLayer(cfg.dropout_rate, 'Name', 'fc_drop')]; %#ok<AGROW>
end
layers = [layers;
    fullyConnectedLayer(1, 'Name', 'fc_out')
    regressionLayer('Name', 'regression')];
end

function options = local_build_cnn_training_options(cfg, model_opts, Xval, Yval)
% 构造 CNN 训练参数，包括验证集、早停和 L2 正则。
options_args = { ...
    'MaxEpochs', cfg.max_epochs, ...
    'MiniBatchSize', cfg.mini_batch_size, ...
    'InitialLearnRate', cfg.initial_learn_rate, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'cpu', ...
    'Plots', 'none', ...
    'ValidationData', {local_matrix_to_sequence_cells(Xval), Yval}, ...
    'ValidationFrequency', 10};
if model_opts.l1_l2_regularization
    options_args = [options_args, {'L2Regularization', model_opts.l2_lambda}];
end
if model_opts.early_stopping
    options_args = [options_args, {'ValidationPatience', model_opts.validation_patience, 'OutputNetwork', 'best-validation-loss'}];
else
    options_args = [options_args, {'ValidationPatience', Inf}];
end
options = trainingOptions('adam', options_args{:});
end

function seq = local_matrix_to_sequence_cells(X)
seq = arrayfun(@(i) reshape(X(i, :), 1, []), 1:size(X, 1), 'UniformOutput', false)';
end

function rmse = local_xgboost_cv_rmse(Xtrain, Ytrain, cfg, nfold)
% 用训练集内部交叉验证评估某组提升树参数的 RMSE。
n = size(Xtrain, 1);
fold_id = local_make_fold_id(n, nfold, 7);
ypred = nan(n, 1);
for f = 1:nfold
    te = fold_id == f;
    tr = ~te;
    try
        tree = templateTree('MaxNumSplits', cfg.max_num_splits, 'MinLeafSize', cfg.min_leaf_size);
        mdl = fitrensemble(Xtrain(tr, :), Ytrain(tr), ...
            'Method', 'LSBoost', ...
            'Learners', tree, ...
            'NumLearningCycles', cfg.num_learning_cycles, ...
            'LearnRate', cfg.learn_rate);
        ypred(te) = predict(mdl, Xtrain(te, :));
    catch
        rmse = inf;
        return;
    end
end
valid = isfinite(Ytrain) & isfinite(ypred);
if nnz(valid) < 2
    rmse = inf;
else
    rmse = sqrt(mean((Ytrain(valid) - ypred(valid)).^2));
end
end

function rmse = local_svr_cv_rmse(Xtrain, Ytrain, cfg, nfold)
n = size(Xtrain, 1);
fold_id = local_make_fold_id(n, nfold, 21);
ypred = nan(n, 1);
for f = 1:nfold
    te = (fold_id == f);
    tr = ~te;
    try
        mdl = fitrsvm(Xtrain(tr, :), Ytrain(tr), ...
            'KernelFunction', cfg.kernel, ...
            'BoxConstraint', cfg.box, ...
            'KernelScale', cfg.scale, ...
            'Standardize', true);
        ypred(te) = predict(mdl, Xtrain(te, :));
    catch
        rmse = inf;
        return;
    end
end
valid = isfinite(Ytrain) & isfinite(ypred);
if nnz(valid) < 2
    rmse = inf;
else
    rmse = sqrt(mean((Ytrain(valid) - ypred(valid)).^2));
end
end

function rmse = local_rf_cv_rmse(Xtrain, Ytrain, cfg, nfold)
n = size(Xtrain, 1);
fold_id = local_make_fold_id(n, nfold, 22);
ypred = nan(n, 1);
for f = 1:nfold
    te = (fold_id == f);
    tr = ~te;
    try
        mdl = TreeBagger(cfg.num_trees, Xtrain(tr, :), Ytrain(tr), ...
            'Method', 'regression', ...
            'MinLeafSize', cfg.min_leaf, ...
            'OOBPrediction', 'off');
        pred_fold = predict(mdl, Xtrain(te, :));
        if iscell(pred_fold), pred_fold = str2double(pred_fold); end
        ypred(te) = pred_fold(:);
    catch
        rmse = inf;
        return;
    end
end
valid = isfinite(Ytrain) & isfinite(ypred);
if nnz(valid) < 2
    rmse = inf;
else
    rmse = sqrt(mean((Ytrain(valid) - ypred(valid)).^2));
end
end

function rmse = local_gpr_cv_rmse(Xtrain, Ytrain, cfg, nfold)
n = size(Xtrain, 1);
fold_id = local_make_fold_id(n, nfold, 23);
ypred = nan(n, 1);
for f = 1:nfold
    te = (fold_id == f);
    tr = ~te;
    try
        mdl = fitrgp(Xtrain(tr, :), Ytrain(tr), ...
            'KernelFunction', cfg.kernel, ...
            'Standardize', true);
        ypred(te) = predict(mdl, Xtrain(te, :));
    catch
        rmse = inf;
        return;
    end
end
valid = isfinite(Ytrain) & isfinite(ypred);
if nnz(valid) < 2
    rmse = inf;
else
    rmse = sqrt(mean((Ytrain(valid) - ypred(valid)).^2));
end
end

function rmse = local_knn_cv_rmse(Xtrain, Ytrain, cfg, nfold)
n = size(Xtrain, 1);
fold_id = local_make_fold_id(n, nfold, 24);
ypred = nan(n, 1);
for f = 1:nfold
    te = (fold_id == f);
    tr = ~te;
    try
        ypred(te) = local_knn_predict(Xtrain(tr, :), Ytrain(tr), Xtrain(te, :), cfg.k, cfg.distance, cfg.weighting, false);
    catch
        rmse = inf;
        return;
    end
end
valid = isfinite(Ytrain) & isfinite(ypred);
if nnz(valid) < 2
    rmse = inf;
else
    rmse = sqrt(mean((Ytrain(valid) - ypred(valid)).^2));
end
end

function ypred = local_knn_predict(Xtrain, Ytrain, Xquery, k, distance_name, weighting, exclude_self)
[idx, dist] = knnsearch(Xtrain, Xquery, 'K', min(k + double(exclude_self), size(Xtrain, 1)), 'Distance', distance_name);
if exclude_self
    idx = idx(:, 2:end);
    dist = dist(:, 2:end);
elseif size(idx, 2) > k
    idx = idx(:, 1:k);
    dist = dist(:, 1:k);
end
if size(idx, 2) > k
    idx = idx(:, 1:k);
    dist = dist(:, 1:k);
end
neighbor_y = Ytrain(idx);
if strcmpi(weighting, 'inverse')
    w = 1 ./ max(dist, 1e-12);
    ypred = sum(neighbor_y .* w, 2) ./ sum(w, 2);
else
    ypred = mean(neighbor_y, 2);
end
ypred = ypred(:);
end

function local_save_regression_plot(plot_path, display_name, result)
fig = figure('Visible', 'off');
hold on;
plot(result.ytest2, result.ypred2, '.', 'MarkerSize', 15);
plot(result.Ytrain2, result.ypred2_train, '.', 'MarkerSize', 15);
line_min = min([result.ytest2(:); result.Ytrain2(:)]);
line_max = max([result.ytest2(:); result.Ytrain2(:)]);
plot(line_min:line_max, line_min:line_max, 'r-', 'LineWidth', 1.2);
hold off;
xlabel('Actual');
ylabel('Predicted');
title(sprintf('%s | R2C=%.3f, R2P=%.3f, RMSEP=%.3f', display_name, result.R2_C, result.R2_P, result.RMSEP));
legend('Test', 'Train', 'Ideal', 'Location', 'best');
saveas(fig, plot_path, 'tiff');
close(fig);
end

function smooth_plot_path = local_save_smooth_overview(group_plot_dir, file_base, X_raw, X_post, meta, save_plot)
% 保存 5 张滤波/预处理过程图：原始、SG、MSC、SNV、最终使用光谱。
smooth_plot_path = '';
if isempty(X_raw) || isempty(X_post)
    return;
end
if nargin < 6
    save_plot = true;
end
if save_plot && ~exist(group_plot_dir, 'dir')
    mkdir(group_plot_dir);
end
if save_plot
    smooth_plot_path = fullfile(group_plot_dir, [file_base '_smooth.tif']);
    smooth_mat_path = fullfile(group_plot_dir, [file_base '_smooth.mat']);
end

preproc_mode = local_meta_text(meta, 'preproc_mode', 'none');
sg_order = local_meta_value(meta, 'sg_order', 3);
sg_window = local_meta_value(meta, 'sg_window', 15);
msc_ref_mode = local_meta_text(meta, 'msc_ref_mode', 'mean');
snv_mode = local_meta_text(meta, 'snv_mode', 'standard');
baseline_zero_mode = local_meta_text(meta, 'baseline_zero_mode', 'none');
baseline_zero_scope = local_meta_text(meta, 'baseline_zero_scope', 'cropped_spectrum');
despike_mode = local_meta_text(meta, 'despike_mode', 'none');

source_feature_count = size(X_raw, 2);
if isfield(meta, 'pre_feature_used_band_idx') && ~isempty(meta.pre_feature_used_band_idx)
    used_band_idx = meta.pre_feature_used_band_idx(:)';
elseif isfield(meta, 'used_band_idx') && ~isempty(meta.used_band_idx)
    used_band_idx = meta.used_band_idx(:)';
else
    used_band_idx = 1:source_feature_count;
end
used_band_idx = used_band_idx(used_band_idx >= 1 & used_band_idx <= source_feature_count);
if isempty(used_band_idx)
    used_band_idx = 1:source_feature_count;
end

x_axis_full = local_load_plot_x_axis(local_project_root(), source_feature_count);
x_axis = x_axis_full(used_band_idx);

work_full = local_apply_despike_split(X_raw, despike_mode);
if strcmpi(strtrim(baseline_zero_scope), 'full_spectrum')
    work_full = local_apply_baseline_zero_split(work_full, baseline_zero_mode);
    apply_cropped_baseline = false;
else
    apply_cropped_baseline = true;
end
work_crop = work_full(:, used_band_idx);
raw_zero = work_crop;
if apply_cropped_baseline
    raw_zero = local_apply_baseline_zero_split(raw_zero, baseline_zero_mode);
end

SG_only = SG(work_crop, sg_order, sg_window);
MSC_only = local_msc_with_ref(work_crop, local_choose_plot_msc_ref(work_crop, msc_ref_mode));
SNV_only = SNV(work_crop, snv_mode);
if apply_cropped_baseline
    SG_only = local_apply_baseline_zero_split(SG_only, baseline_zero_mode);
    MSC_only = local_apply_baseline_zero_split(MSC_only, baseline_zero_mode);
    SNV_only = local_apply_baseline_zero_split(SNV_only, baseline_zero_mode);
end

fig = figure('Visible', 'on', 'Color', 'w');
subplot(2,3,1); plot(x_axis, raw_zero'); title('source data');
subplot(2,3,2); plot(x_axis, SG_only'); title('SG only');
subplot(2,3,3); plot(x_axis, MSC_only'); title(['MSC only (' msc_ref_mode ')']);
subplot(2,3,4); plot(x_axis, SNV_only'); title(['SNV only (' snv_mode ')']);
subplot(2,3,5); plot(x_axis, X_post'); title(['final used: ' preproc_mode]);
subplot(2,3,6); axis off;
if save_plot
    saveas(fig, smooth_plot_path, 'tiff');
end
drawnow;
pause(0.8);
close(fig);

if save_plot
    Post_smooth_data = X_post; %#ok<NASGU>
    save(smooth_mat_path, 'Post_smooth_data', 'preproc_mode', 'sg_order', 'sg_window', ...
        'msc_ref_mode', 'snv_mode', 'used_band_idx', 'x_axis', ...
        'baseline_zero_mode', 'baseline_zero_scope', 'despike_mode');
end
end

function feature_plot_path = local_save_selected_band_overlay(group_plot_dir, file_base, X_post, meta, save_plot)
% 保存特征筛选波段图：左图第一阶段筛选，右图第二阶段降维高贡献波段。
feature_plot_path = '';
if ~isfield(meta, 'stage1_used_band_idx') && ~isfield(meta, 'final_used_band_idx')
    return;
end
if nargin < 5
    save_plot = true;
end
if save_plot && ~exist(group_plot_dir, 'dir')
    mkdir(group_plot_dir);
end
if save_plot
    feature_plot_path = fullfile(group_plot_dir, [file_base '_selected_bands.tif']);
end

if isfield(meta, 'full_preprocessed_mean_curve') && ~isempty(meta.full_preprocessed_mean_curve)
    mean_curve = meta.full_preprocessed_mean_curve(:)';
else
    mean_curve = mean(X_post, 1, 'omitnan');
end
source_feature_count = local_meta_value(meta, 'source_feature_count', numel(mean_curve));
x_axis_full = local_load_plot_x_axis(local_project_root(), source_feature_count);
if isfield(meta, 'pre_feature_used_band_idx') && ~isempty(meta.pre_feature_used_band_idx)
    plotted_band_idx = meta.pre_feature_used_band_idx(:)';
else
    plotted_band_idx = 1:numel(mean_curve);
end
plotted_band_idx = plotted_band_idx(plotted_band_idx >= 1 & plotted_band_idx <= source_feature_count);
if numel(plotted_band_idx) ~= numel(mean_curve)
    plotted_band_idx = 1:numel(mean_curve);
    source_feature_count = numel(mean_curve);
    x_axis_full = 1:source_feature_count;
end
full_mean_curve = nan(1, source_feature_count);
full_mean_curve(plotted_band_idx) = mean_curve;
x_axis = x_axis_full(:)';

stage1_idx = local_band_to_plot_positions(local_meta_array(meta, 'stage1_used_band_idx'), plotted_band_idx);
second_idx = local_band_to_plot_positions(local_meta_array(meta, 'second_stage_used_band_idx'), plotted_band_idx);
stage1_band_values = local_meta_array(meta, 'stage1_used_band_idx');
stage1_band_values = stage1_band_values(stage1_band_values >= 1 & stage1_band_values <= numel(full_mean_curve));
stage1_mean_curve = full_mean_curve(stage1_band_values);
stage1_x_axis = x_axis(stage1_band_values);
second_on_stage1_idx = local_band_to_plot_positions(local_meta_array(meta, 'second_stage_used_band_idx'), stage1_band_values);

fig = figure('Visible', 'on', 'Color', 'w');
subplot(1,3,1);
plot(x_axis, full_mean_curve, 'Color', [0.12 0.32 0.72], 'LineWidth', 1.1); hold on;
if ~isempty(stage1_idx)
    stage1_band_idx = plotted_band_idx(stage1_idx);
    scatter(x_axis(stage1_band_idx), full_mean_curve(stage1_band_idx), 24, [0.95 0.45 0.10], 'filled');
end
hold off;
xlabel('Wavelength / Feature');
ylabel('Intensity');
title(sprintf('Full Spectrum + Stage 1 | n=%d', numel(stage1_idx)));
legend('Filtered Mean Spectrum', 'Stage-1 Selected Bands', 'Location', 'best');

subplot(1,3,2);
plot(x_axis, full_mean_curve, 'Color', [0.12 0.32 0.72], 'LineWidth', 1.1); hold on;
if ~isempty(second_idx)
    second_band_idx = plotted_band_idx(second_idx);
    scatter(x_axis(second_band_idx), full_mean_curve(second_band_idx), 30, [0.85 0.10 0.12], 'filled');
end
hold off;
xlabel('Wavelength / Feature');
ylabel('Intensity');
fs_method = upper(local_meta_text(meta, 'fs_method', 'UNKNOWN'));
preproc_mode = local_meta_text(meta, 'preproc_mode', 'unknown');
title(sprintf('Full Spectrum + Stage 2 | n=%d', numel(second_idx)));
second_note = local_meta_text(meta, 'second_stage_note', 'Second-stage selected/contributing bands');
legend('Filtered Mean Spectrum', second_note, 'Location', 'best');

subplot(1,3,3);
if isempty(stage1_x_axis)
    plot(x_axis, full_mean_curve, 'Color', [0.12 0.32 0.72], 'LineWidth', 1.1); hold on;
else
    plot(stage1_x_axis, stage1_mean_curve, 'Color', [0.12 0.32 0.72], 'LineWidth', 1.1); hold on;
end
if ~isempty(second_on_stage1_idx) && ~isempty(stage1_x_axis)
    scatter(stage1_x_axis(second_on_stage1_idx), stage1_mean_curve(second_on_stage1_idx), 30, [0.85 0.10 0.12], 'filled');
elseif ~isempty(second_idx)
    second_band_idx = plotted_band_idx(second_idx);
    scatter(x_axis(second_band_idx), full_mean_curve(second_band_idx), 30, [0.85 0.10 0.12], 'filled');
end
hold off;
xlabel('Wavelength / Feature');
ylabel('Intensity');
title(sprintf('Stage 2 on Stage 1 | n=%d', max(numel(second_on_stage1_idx), numel(second_idx))));
legend('Filtered Mean Spectrum', second_note, 'Location', 'best');
sgtitle(sprintf('Feature Selection | %s | preproc=%s', fs_method, preproc_mode));
if save_plot
    saveas(fig, feature_plot_path, 'tiff');
end
drawnow;
pause(0.8);
close(fig);
end

function save_plot = local_should_save_smooth_feature_plots(generalization_options)
save_plot = true;
if isfield(generalization_options, 'evaluation') && ...
        isfield(generalization_options.evaluation, 'save_smooth_and_feature_plots') && ...
        ~isempty(generalization_options.evaluation.save_smooth_and_feature_plots)
    save_plot = logical(generalization_options.evaluation.save_smooth_and_feature_plots);
end
end

function project_root = local_project_root()
project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end

function file_base = local_make_output_file_base(meta, method, time_tag)
fs_method = local_meta_text(meta, 'fs_method', 'none');
model_method = local_get_model_display_name(method);
file_base = sprintf('%s_%s_%s', upper(fs_method), upper(model_method), time_tag);
file_base = regexprep(file_base, '[^\w]+', '_');
file_base = regexprep(file_base, '_+', '_');
file_base = strtrim(file_base);
end

function group_plot_dir = local_make_group_plot_dir(project_root, meta, method, file_base)
% Smooth 结果目录层级：
% Result/Smooth/特征方法/模型算法/特征方法_模型算法_时间/
fs_method = local_safe_path_part(upper(local_meta_text(meta, 'fs_method', 'none')));
model_method = local_safe_path_part(upper(local_get_model_display_name(method)));
group_name = local_safe_path_part(file_base);
group_plot_dir = fullfile(project_root, 'Result', 'Smooth', fs_method, model_method, group_name);
end

function part = local_safe_path_part(value)
part = char(string(value));
part = regexprep(part, '[^\w]+', '_');
part = regexprep(part, '_+', '_');
part = strtrim(part);
if isempty(part)
    part = 'UNKNOWN';
end
end

function tag = local_r2p_to_percent_tag(r2p)
if ~isfinite(r2p)
    tag = 'NaN';
else
    tag = sprintf('%d', round(r2p * 100));
end
end

function tag = local_now_file_time_tag()
t = now;
millisecond = round(mod(t * 86400000, 1000));
tag = sprintf('%s_%03d', datestr(t, 'yyyymmdd_HHMMSS'), millisecond);
end

function ref = local_choose_plot_msc_ref(Xin, reference_mode)
switch lower(strtrim(reference_mode))
    case 'mean'
        ref = mean(Xin, 1, 'omitnan');
    case 'median'
        ref = median(Xin, 1, 'omitnan');
    case 'first'
        ref = Xin(1, :);
    otherwise
        ref = mean(Xin, 1, 'omitnan');
end
end

function x_axis = local_load_plot_x_axis(project_root, feature_count)
wave_mat = fullfile(project_root, 'data', 'wavelength144.mat');
if exist(wave_mat, 'file')
    S = load(wave_mat);
    names = fieldnames(S);
    for i = 1:numel(names)
        v = S.(names{i});
        if isnumeric(v) && numel(v) == feature_count
            x_axis = v(:)';
            return;
        end
    end
end
nir_dir = fullfile(project_root, 'data', 'NIR');
files = dir(fullfile(nir_dir, '*.csv'));
if ~isempty(files)
    raw = readmatrix(fullfile(nir_dir, files(1).name));
    if size(raw, 1) == feature_count && size(raw, 2) >= 1
        x_axis = raw(:, 1)';
        return;
    end
end
x_axis = 1:feature_count;
end

function txt = local_meta_text(meta, field_name, default_value)
if isfield(meta, field_name) && ~isempty(meta.(field_name))
    txt = char(string(meta.(field_name)));
else
    txt = default_value;
end
end

function value = local_meta_value(meta, field_name, default_value)
if isfield(meta, field_name) && ~isempty(meta.(field_name))
    value = meta.(field_name);
else
    value = default_value;
end
end

function values = local_meta_array(meta, field_name)
if isfield(meta, field_name) && ~isempty(meta.(field_name))
    values = meta.(field_name)(:)';
else
    values = [];
end
end

function pos = local_band_to_plot_positions(band_idx, plotted_band_idx)
if isempty(band_idx) || isempty(plotted_band_idx)
    pos = [];
    return;
end
[tf, loc] = ismember(band_idx(:)', plotted_band_idx(:)');
pos = unique(loc(tf), 'stable');
pos = pos(pos >= 1 & pos <= numel(plotted_band_idx));
end

function result = build_result_struct(method_name, A, selected_info, select_idx, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, CV)
Ytrain = double(Ytrain(:));
Ytest = double(Ytest(:));
Ypred_train = double(Ypred_train(:));
Ypred = double(Ypred(:));

valid_train = isfinite(Ytrain) & isfinite(Ypred_train);
valid_test = isfinite(Ytest) & isfinite(Ypred);
R2_C = NaN; R2_P = NaN; RMSEC = NaN; RMSEP = NaN; RPD = NaN;
if nnz(valid_train) >= 2
    ytr = Ytrain(valid_train);
    yptr = Ypred_train(valid_train);
    sst = sum((ytr - mean(ytr)).^2);
    sse = sum((ytr - yptr).^2);
    RMSEC = sqrt(mean((ytr - yptr).^2));
    if sst > eps, R2_C = 1 - sse / sst; end
end
if nnz(valid_test) >= 2
    yte = Ytest(valid_test);
    ypte = Ypred(valid_test);
    sst = sum((yte - mean(yte)).^2);
    sse = sum((yte - ypte).^2);
    RMSEP = sqrt(mean((yte - ypte).^2));
    if sst > eps, R2_P = 1 - sse / sst; end
    if RMSEP > 0, RPD = std(yte) / RMSEP; end
end
result = struct();
result.method_name = method_name;
result.A = A;
result.selected_info = selected_info;
result.Rank2 = select_idx;
result.Xtrain2 = Xtrain;
result.Ytrain2 = Ytrain;
result.Xtest2 = Xtest;
result.ytest2 = Ytest;
result.ypred2_train = Ypred_train;
result.ypred2 = Ypred;
result.R2_C = R2_C;
result.R2_P = R2_P;
result.RMSEC = RMSEC;
result.RMSEP = RMSEP;
result.RPD = RPD;
if isempty(CV)
    result.RMSECV = NaN;
    result.Q2_Max = NaN;
else
    result.RMSECV = CV.RMSECV_min;
    result.Q2_Max = CV.Q2_max;
end
end

function rmse = local_pcr_cv_rmse(score_train, y_train, nfold)
y_train = double(y_train(:));
n = size(score_train, 1);
fold_id = local_make_fold_id(n, nfold, 1);
ypred = nan(n, 1);
for f = 1:nfold
    te = fold_id == f;
    tr = ~te;
    beta = [ones(nnz(tr), 1), score_train(tr, :)] \ y_train(tr);
    ypred(te) = [ones(nnz(te), 1), score_train(te, :)] * beta;
end
valid = isfinite(y_train) & isfinite(ypred);
if nnz(valid) < 2
    rmse = inf;
else
    rmse = sqrt(mean((y_train(valid) - ypred(valid)).^2));
end
end

function fold_id = local_make_fold_id(n, nfold, seed)
nfold = max(2, min(nfold, n));
rng(seed);
perm = randperm(n);
fold_id = zeros(n, 1);
for i = 1:n
    fold_id(perm(i)) = mod(i - 1, nfold) + 1;
end
end

function [Xtrain_out, Xtest_out, meta] = local_preprocess_split(Xtrain_in, Xtest_in, meta)
Xtrain_out = Xtrain_in;
Xtest_out = Xtest_in;
if nargin < 3 || isempty(meta)
    return;
end
cut_left = 0;
cut_right = 0;
if isfield(meta, 'cut_left') && ~isempty(meta.cut_left), cut_left = meta.cut_left; end
if isfield(meta, 'cut_right') && ~isempty(meta.cut_right), cut_right = meta.cut_right; end
source_feature_count = size(Xtrain_in, 2);
used_band_idx = (1 + cut_left):(source_feature_count - cut_right);
if isempty(used_band_idx)
    error('No features remain after cropping.');
end
baseline_zero_mode = 'none';
if isfield(meta, 'baseline_zero_mode') && ~isempty(meta.baseline_zero_mode), baseline_zero_mode = meta.baseline_zero_mode; end
despike_mode = 'none';
if isfield(meta, 'despike_mode') && ~isempty(meta.despike_mode), despike_mode = meta.despike_mode; end
baseline_zero_scope = 'cropped_spectrum';
if isfield(meta, 'baseline_zero_scope') && ~isempty(meta.baseline_zero_scope), baseline_zero_scope = meta.baseline_zero_scope; end
preproc_mode = 'none';
if isfield(meta, 'preproc_mode') && ~isempty(meta.preproc_mode), preproc_mode = meta.preproc_mode; end
sg_order = 3;
if isfield(meta, 'sg_order') && ~isempty(meta.sg_order), sg_order = meta.sg_order; end
sg_window = 15;
if isfield(meta, 'sg_window') && ~isempty(meta.sg_window), sg_window = meta.sg_window; end
msc_ref_mode = 'mean';
if isfield(meta, 'msc_ref_mode') && ~isempty(meta.msc_ref_mode), msc_ref_mode = meta.msc_ref_mode; end
snv_mode = 'standard';
if isfield(meta, 'snv_mode') && ~isempty(meta.snv_mode), snv_mode = meta.snv_mode; end

Xtrain_work = Xtrain_in;
Xtest_work = Xtest_in;
if strcmpi(strtrim(baseline_zero_scope), 'full_spectrum')
    Xtrain_work = local_apply_despike_split(Xtrain_work, despike_mode);
    Xtest_work = local_apply_despike_split(Xtest_work, despike_mode);
    Xtrain_work = local_apply_baseline_zero_split(Xtrain_work, baseline_zero_mode);
    Xtest_work = local_apply_baseline_zero_split(Xtest_work, baseline_zero_mode);
    Xtrain_work = Xtrain_work(:, used_band_idx);
    Xtest_work = Xtest_work(:, used_band_idx);
else
    Xtrain_work = Xtrain_work(:, used_band_idx);
    Xtest_work = Xtest_work(:, used_band_idx);
    Xtrain_work = local_apply_despike_split(Xtrain_work, despike_mode);
    Xtest_work = local_apply_despike_split(Xtest_work, despike_mode);
    Xtrain_work = local_apply_baseline_zero_split(Xtrain_work, baseline_zero_mode);
    Xtest_work = local_apply_baseline_zero_split(Xtest_work, baseline_zero_mode);
end

[Xtrain_out, Xtest_out] = local_apply_preproc_pair(Xtrain_work, Xtest_work, preproc_mode, sg_order, sg_window, msc_ref_mode, snv_mode);
meta.used_band_idx = used_band_idx;
meta.used_band_range = [used_band_idx(1), used_band_idx(end)];
end

function [Xtrain_out, Xtest_out] = local_apply_preproc_pair(Xtrain_in, Xtest_in, preproc_mode, sg_order, sg_window, msc_ref_mode, snv_mode)
mode = lower(strtrim(preproc_mode));
switch mode
    case 'none'
        Xtrain_out = Xtrain_in;
        Xtest_out = Xtest_in;
    case 'sg'
        Xtrain_out = SG(Xtrain_in, sg_order, sg_window);
        Xtest_out = SG(Xtest_in, sg_order, sg_window);
    case 'sg+snv'
        Xtrain_out = SNV(SG(Xtrain_in, sg_order, sg_window), snv_mode);
        Xtest_out = SNV(SG(Xtest_in, sg_order, sg_window), snv_mode);
    case 'sg+msc+snv'
        Xtrain_sg = SG(Xtrain_in, sg_order, sg_window);
        Xtest_sg = SG(Xtest_in, sg_order, sg_window);
        [Xtrain_msc, Xtest_msc] = local_apply_msc_pair(Xtrain_sg, Xtest_sg, msc_ref_mode);
        Xtrain_out = SNV(Xtrain_msc, snv_mode);
        Xtest_out = SNV(Xtest_msc, snv_mode);
    otherwise
        Xtrain_sg = SG(Xtrain_in, sg_order, sg_window);
        Xtest_sg = SG(Xtest_in, sg_order, sg_window);
        [Xtrain_out, Xtest_out] = local_apply_msc_pair(Xtrain_sg, Xtest_sg, msc_ref_mode);
end
end

function [Xtrain_msc, Xtest_msc] = local_apply_msc_pair(Xtrain_in, Xtest_in, reference_mode)
switch lower(strtrim(reference_mode))
    case 'mean'
        ref = mean(Xtrain_in, 1, 'omitnan');
    case 'median'
        ref = median(Xtrain_in, 1, 'omitnan');
    case 'first'
        ref = Xtrain_in(1, :);
    otherwise
        ref = mean(Xtrain_in, 1, 'omitnan');
end
Xtrain_msc = local_msc_with_ref(Xtrain_in, ref);
Xtest_msc = local_msc_with_ref(Xtest_in, ref);
end

function Xout = local_msc_with_ref(Xin, ref)
Xout = zeros(size(Xin));
for i = 1:size(Xin, 1)
    row = Xin(i, :);
    valid = isfinite(ref) & isfinite(row);
    if nnz(valid) < 2
        Xout(i, :) = row;
        continue;
    end
    p = polyfit(ref(valid), row(valid), 1);
    if ~isfinite(p(1)) || abs(p(1)) < 1e-8
        Xout(i, :) = row - p(2);
    else
        Xout(i, :) = (row - p(2)) ./ p(1);
    end
end
end

function X_out = local_apply_baseline_zero_split(X_in, mode)
X_out = X_in;
if isempty(X_in)
    return;
end
switch lower(strtrim(mode))
    case 'none'
        return;
    case 'first_point'
        base = X_in(:, 1);
    case 'first_5_mean'
        base = mean(X_in(:, 1:min(5, size(X_in, 2))), 2, 'omitnan');
    otherwise
        error('Unsupported baseline mode: %s', mode);
end
X_out = X_in - base;
end

function X_out = local_apply_despike_split(X_in, mode)
X_out = X_in;
if isempty(X_in)
    return;
end
switch lower(strtrim(mode))
    case 'none'
        return;
    case 'median3'
        X_out = medfilt1(X_in, 3, [], 2, 'truncate');
    case 'median5'
        X_out = medfilt1(X_in, 5, [], 2, 'truncate');
    case 'median7'
        X_out = medfilt1(X_in, 7, [], 2, 'truncate');
    otherwise
        X_out = local_jump_guard_split(X_in);
end
end

function X_out = local_jump_guard_split(X_in)
X_out = X_in;
if size(X_in, 2) < 3
    return;
end
for r = 1:size(X_in, 1)
    row = X_in(r, :);
    for c = 2:(numel(row) - 1)
        left_v = row(c - 1);
        mid_v = row(c);
        right_v = row(c + 1);
        if abs(mid_v - left_v) > 4 * max(abs(left_v - right_v), eps) && abs(mid_v - right_v) > 4 * max(abs(left_v - right_v), eps)
            row(c) = (left_v + right_v) / 2;
        end
    end
    X_out(r, :) = row;
end
end

function name = local_get_model_display_name(method)
switch lower(strtrim(method))
    case 'pls'
        name = 'PLS';
    case 'pcr'
        name = 'PCR';
    case 'svr'
        name = 'SVR';
    case 'rf'
        name = 'RF';
    case 'gpr'
        name = 'GPR';
    case 'knn'
        name = 'KNN';
    case 'xgboost'
        name = 'XGBoost';
    case 'cnn'
        name = 'CNN';
    otherwise
        name = upper(method);
end
end

function suffix = ternary_suffix(cond, value)
if cond
    suffix = value;
else
    suffix = '';
end
end

function opts = default_generalization_options_local()
opts = struct();
opts.feature = struct( ...
    'nested_cv_selection', false, ...
    'inner_cv_folds', 5, ...
    'use_stable_features', false, ...
    'stability_threshold', 0.60, ...
    'post_feature_pca_projection', false, ...
    'post_feature_pca_components', 20);
opts.data_processing = struct('data_augmentation', false, 'augmentation_copies', 1, 'noise_std', 0.003, 'max_shift', 2);
opts.validation = struct('use_external_holdout', false, 'external_ratio', 0.20, 'random_seed', 42);
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
    'xgboost_simplify', false, ...
    'xgb_max_depth_cap', 4, ...
    'xgb_learn_rate_cap', 0.03, ...
    'xgb_min_leaf_floor', 8, ...
    'xgboost_grid_mode', 'quick', ...
    'cnn_simplify', false, ...
    'cnn_max_blocks', 2, ...
    'cnn_fc_units_cap', 48, ...
    'cnn_grid_mode', 'quick');
opts.evaluation = struct('enforce_small_rc_rp_gap', false, 'rc_rp_gap_threshold', 0.05, 'save_smooth_and_feature_plots', true);
end

function opts = normalize_generalization_options_local(opts)
opts = merge_struct_recursive_local(default_generalization_options_local(), opts);
end

function out = merge_struct_recursive_local(base, override)
out = base;
if ~isstruct(override)
    return;
end
fields = fieldnames(override);
for i = 1:numel(fields)
    key = fields{i};
    if isfield(base, key) && isstruct(base.(key)) && isstruct(override.(key))
        out.(key) = merge_struct_recursive_local(base.(key), override.(key));
    else
        out.(key) = override.(key);
    end
end
end
