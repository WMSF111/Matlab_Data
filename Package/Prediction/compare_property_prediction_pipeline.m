function summary = compare_property_prediction_pipeline(property_name, filter_method, feature_selection_method, sg_order, sg_window, include_preprocessed_group, generalization_options)
% Compare property prediction performance with optional generalization controls.

if nargin < 1 || isempty(property_name)
    error('property_name is required.');
end
if nargin < 2 || isempty(filter_method)
    filter_method = 'sg+msc+snv';
end
if nargin < 3 || isempty(feature_selection_method)
    feature_selection_method = 'pca';
end
if nargin < 4 || isempty(sg_order)
    sg_order = 3;
end
if nargin < 5 || isempty(sg_window)
    sg_window = 35;
end
if nargin < 6 || isempty(include_preprocessed_group)
    include_preprocessed_group = false;
end
if nargin < 7 || isempty(generalization_options)
    generalization_options = default_generalization_options();
else
    generalization_options = normalize_generalization_options(generalization_options);
end

% 支持一次输入多个特征方法；这里会递归展开后再汇总结果。
if iscell(feature_selection_method) || (isstring(feature_selection_method) && numel(feature_selection_method) > 1)
    method_list = cellstr(feature_selection_method);
    summary_parts = cell(numel(method_list), 1);
    for ii = 1:numel(method_list)
        summary_parts{ii} = compare_property_prediction_pipeline(property_name, filter_method, method_list{ii}, sg_order, sg_window, include_preprocessed_group, generalization_options);
    end
    summary = vertcat(summary_parts{:});
    return;
end

% 清理命令窗口和图窗，避免上一次运行的可视化干扰本次比较。
clc;
close all;

% 自动定位项目根目录并加入路径，避免手动维护 addpath。
project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(project_root));

% 最终预处理链只由 filter_method 决定，不再通过额外开关改写。
preproc_mode = lower(strtrim(filter_method));
fs_method = lower(strtrim(char(feature_selection_method)));
msc_ref_mode = 'mean';
snv_mode = 'robust';
keep_dataset_exports = false;
baseline_zero_mode = 'first_5_mean';
baseline_zero_scope = 'full_spectrum';
despike_mode = 'jump_guard';

% 特征选择参数网格：这里统一定义不同方法的候选参数。
feature_selection_param_grid = struct();
% 第一阶段特征筛选默认保留 80~200 个候选特征；
% 如果开启 post_feature_pca_projection，后面还会再压到 shared_component_cap。
% CARS 中 num_sampling_runs 决定采样强度，target_count 决定最终最多保留多少维。
feature_selection_param_grid.cars = { ...
    struct('num_sampling_runs', 60, 'target_count', 60)};
feature_selection_param_grid.pca = {80, 120, 160, 200};
feature_selection_param_grid.corr_topk = {80, 120, 160, 200};
feature_selection_param_grid.spa = {80, 120, 160, 200};

% 回归器候选集合：统一在这里管理参与比较的模型。
regressors = {'pls', 'pcr', 'svr', 'rf', 'gpr', 'knn', 'xgboost', 'cnn'};
% regressors = {'xgboost', 'cnn'};
regressor_params = struct();
regressor_params.pls = {struct('max_lv', 300, 'cv_fold', 10)};
regressor_params.pcr = {struct('max_pc', 300)};
regressor_params.svr = {struct('kernels', {{'linear', 'gaussian'}}, 'boxes', [0.1 0.5 1 5], 'scales', {{'auto', 0.5, 1, 2, 5}})};
regressor_params.rf = {struct('num_trees', [100 200], 'min_leaf', [1 5 10 20])};
regressor_params.gpr = {struct('kernels', {{'squaredexponential', 'matern32', 'matern52'}})};
regressor_params.knn = {struct('k', [1 3 5 7 9 11 15], 'distances', {{'euclidean', 'cityblock', 'cosine', 'correlation'}}, 'weightings', {{'uniform', 'inverse'}})};
% XGBoost/CNN 的参数网格支持 quick/full/off 三档切换。
regressor_params.xgboost = build_xgboost_param_grid(generalization_options.model);
regressor_params.cnn = build_cnn_param_grid(generalization_options.model);
regressor_params = apply_model_complexity_switches(regressor_params, generalization_options.model);

if ~isfield(feature_selection_param_grid, fs_method)
    error('Unsupported feature selection method: %s', fs_method);
end
fs_param_grid = feature_selection_param_grid.(fs_method);

% 为本次运行生成唯一标签，便于保存结果和回溯配置。
run_tag = datestr(now, 'yyyymmdd_HHMMSS');
setenv('HXR_RUN_TAG', run_tag);

property_tag = property_to_tag(property_name);
summary_dir = fullfile(project_root, 'Result', 'Summary', [property_tag '_regressor_compare_' run_tag]);
if ~exist(summary_dir, 'dir')
    mkdir(summary_dir);
end

% 在训练前先检查目标值分布和缺失情况，减少无效运行。
diagnose_property_targets(project_root, property_name);

step_total = 0;
% 阶段 1：仅做预处理，不做特征筛选，作为对照组。
if include_preprocessed_group
    for i = 1:numel(regressors)
        step_total = step_total + numel(regressor_params.(regressors{i}));
    end
end
for i = 1:numel(fs_param_grid)
    for j = 1:numel(regressors)
        step_total = step_total + numel(regressor_params.(regressors{j}));
    end
end

fprintf('================ Pipeline Compare Start ================\n');
fprintf('Property: %s\n', property_name);
fprintf('Preprocess: %s\n', upper(preproc_mode));
fprintf('Feature method: %s\n', upper(fs_method));
fprintf('Generalization switches: %s\n', generalization_options_to_text(generalization_options));
fprintf('Summary dir: %s\n', summary_dir);
fprintf('Total combinations: %d\n', step_total);
step_id = 0;
all_rows = {};
temp_dataset_dirs = {};
global_tic = tic;
best_tracker = struct('initialized', false, 'r2p', -inf, 'rmsep', inf, 'summary', '');

if include_preprocessed_group
    fprintf('\n[Stage 1] Preparing preprocessed dataset...\n');
    dataset_pre = prepare_property_dataset(property_name, 'preprocessed', preproc_mode, sg_order, sg_window, 'corr_topk', [], msc_ref_mode, snv_mode, keep_dataset_exports, baseline_zero_mode, despike_mode, baseline_zero_scope);
    temp_dataset_dirs{end + 1} = dataset_pre.paths.dir;
    for i = 1:numel(regressors)
        method_name = regressors{i};
        params = regressor_params.(method_name);
        for j = 1:numel(params)
            step_id = step_id + 1;
            method_param = params{j};
            fprintf('\n[Progress %d/%d] preprocessed | model=%s | param=%s\n', step_id, step_total, get_model_display_name(method_name), param_to_text(method_param));
            result = train_model_with_generalization(dataset_pre.paths.mat, method_name, method_param, generalization_options);
            print_eta(step_id, step_total, global_tic);
            all_rows(end + 1, :) = build_report_row('仅滤波', 'none', '', get_model_display_name(method_name), param_to_text(method_param), result, generalization_options); %#ok<AGROW>
            best_tracker = print_best_update_if_needed(best_tracker, '仅滤波', 'none', '', get_model_display_name(method_name), param_to_text(method_param), result);
        end
    end
end

% 阶段 2：预处理后再做特征选择，是主要比较组。
fprintf('\n[Stage 2] Preparing selected dataset...\n');
for i = 1:numel(fs_param_grid)
    fs_param = fs_param_grid{i};
    fprintf('\n---- FS=%s | param=%s ----\n', upper(fs_method), param_to_text(fs_param));
    dataset_sel = prepare_property_dataset(property_name, 'selected', preproc_mode, sg_order, sg_window, fs_method, fs_param, msc_ref_mode, snv_mode, keep_dataset_exports, baseline_zero_mode, despike_mode, baseline_zero_scope);
    temp_dataset_dirs{end + 1} = dataset_sel.paths.dir;
    for j = 1:numel(regressors)
        method_name = regressors{j};
        params = regressor_params.(method_name);
        for k = 1:numel(params)
            step_id = step_id + 1;
            method_param = params{k};
            fprintf('\n[Progress %d/%d] selected | %s=%s | model=%s | param=%s\n', step_id, step_total, upper(fs_method), param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param));
            result = train_model_with_generalization(dataset_sel.paths.mat, method_name, method_param, generalization_options);
            print_eta(step_id, step_total, global_tic);
            all_rows(end + 1, :) = build_report_row('特征筛选后', fs_method, param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param), result, generalization_options); %#ok<AGROW>
            best_tracker = print_best_update_if_needed(best_tracker, '特征筛选后', fs_method, param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param), result);
        end
    end
end

% 将全部实验结果整理成表格并按泛化表现排序保存。
all_results_table = cell2table(all_rows, 'VariableNames', report_var_names());
all_results_table = sortrows(all_results_table, {'数据组名称', '回归器名称', 'R2_P', 'RMSEP'}, {'ascend', 'ascend', 'descend', 'ascend'});
summary_table = build_best_group_model_table(all_results_table);
save(fullfile(summary_dir, 'summary.mat'), 'all_results_table', 'summary_table', 'generalization_options');
writetable(all_results_table, fullfile(summary_dir, 'all_results.csv'));
writetable(summary_table, fullfile(summary_dir, 'best_models.csv'));
write_param_details_file(summary_dir, all_results_table, summary_table);

cleanup_temp_datasets(temp_dataset_dirs);
cleanup_temp_runtime(project_root);
setenv('HXR_RUN_TAG', '');

summary = all_results_table;

fprintf('\n================ Pipeline Compare Complete ================\n');
fprintf('Elapsed: %s\n', format_duration(toc(global_tic)));
fprintf('Saved to: %s\n', summary_dir);
end

function row = build_report_row(group_name, fs_method, fs_param_text, model_display_name, train_param_text, result, generalization_options)
% 将一次模型训练结果整理成汇总表的一行，便于后续横向比较。
meta = result.dataset_metadata;
used_band_range_text = '';
used_feature_count = NaN;
if isfield(meta, 'used_band_range') && numel(meta.used_band_range) >= 2
    used_band_range_text = sprintf('%d:%d', meta.used_band_range(1), meta.used_band_range(end));
end
if isfield(meta, 'final_feature_count') && ~isempty(meta.final_feature_count)
    used_feature_count = meta.final_feature_count;
elseif isfield(meta, 'used_band_idx') && ~isempty(meta.used_band_idx)
    used_feature_count = numel(meta.used_band_idx);
elseif isfield(meta, 'X') && ~isempty(meta.X)
    used_feature_count = size(meta.X, 2);
end
rc_rp_gap = abs(result.R2_C - result.R2_P);
if generalization_options.evaluation.enforce_small_rc_rp_gap
    gap_flag = ternary_text(rc_rp_gap <= generalization_options.evaluation.rc_rp_gap_threshold, '通过', '风险');
else
    gap_flag = ternary_text(rc_rp_gap <= generalization_options.evaluation.rc_rp_gap_threshold, '正常', '偏大');
end
row = {group_name, fs_method, fs_param_text, model_display_name, train_param_text, result.best_param_detail, ...
    meta.preproc_mode, meta.sg_order, meta.sg_window, meta.msc_ref_mode, meta.snv_mode, meta.baseline_zero_mode, meta.despike_mode, ...
    used_band_range_text, used_feature_count, ...
    result.R2_C, result.R2_P, rc_rp_gap, gap_flag, result.RMSEC, result.RMSEP, result.RPD, ...
    local_result_field(result, 'stage1_R2_C', NaN), local_result_field(result, 'stage1_R2_P', NaN), ...
    local_result_field(result, 'stage1_RMSEC', NaN), local_result_field(result, 'stage1_RMSEP', NaN), local_result_field(result, 'stage1_RPD', NaN), ...
    local_result_field(result, 'stage2_R2_C', result.R2_C), local_result_field(result, 'stage2_R2_P', result.R2_P), ...
    local_result_field(result, 'stage2_RMSEC', result.RMSEC), local_result_field(result, 'stage2_RMSEP', result.RMSEP), local_result_field(result, 'stage2_RPD', result.RPD), ...
    ternary_text(result.used_external_validation, '开启', '关闭'), ...
    ternary_text(result.used_nested_cv_selection, '开启', '关闭'), ...
    ternary_text(result.used_stable_feature_selection, '开启', '关闭'), ...
    ternary_text(result.used_data_augmentation, '开启', '关闭'), ...
    generalization_options_to_text(generalization_options), result.result_mat_path, result.regression_plot_path, ...
    local_result_field(result, 'stage1_result_mat_path', ''), local_result_field(result, 'stage2_result_mat_path', ''), ...
    local_result_field(result, 'stage1_regression_plot_path', ''), local_result_field(result, 'stage2_regression_plot_path', '')};
end

function names = report_var_names()
names = {'数据组名称', '特征筛选方法', '特征筛选参数', '回归器名称', '输入参数', '最优参数详情', ...
    '预处理方式', 'SG阶数', 'SG窗口', 'MSC模式', 'SNV模式', '基线归零模式', '去尖刺模式', ...
    '使用波段范围', '最终特征维数', 'R2_C', 'R2_P', 'R2差值_RC_RP', 'RC_RP差值判断', 'RMSEC', 'RMSEP', 'RPD', ...
    '第一次R2_C', '第一次R2_P', '第一次RMSEC', '第一次RMSEP', '第一次RPD', ...
    '第二次R2_C', '第二次R2_P', '第二次RMSEC', '第二次RMSEP', '第二次RPD', ...
    '外部验证', '嵌套CV特征选择', '稳定特征筛选', '数据增强', '泛化优化配置', '模型结果MAT路径', '回归图像路径', ...
    '第一次模型MAT路径', '第二次模型MAT路径', '第一次回归图像路径', '第二次回归图像路径'};
end

function value = local_result_field(result, field_name, default_value)
if isfield(result, field_name) && ~isempty(result.(field_name))
    value = result.(field_name);
else
    value = default_value;
end
end

function summary_table = build_best_group_model_table(all_results_table)
% 从全量结果里挑出每个“数据组-模型”组合下表现最好的记录。
rows = {};
group_col = all_results_table{:, '数据组名称'};
model_col = all_results_table{:, '回归器名称'};
group_names = unique(group_col, 'stable');
model_names = unique(model_col, 'stable');
for i = 1:numel(group_names)
    for j = 1:numel(model_names)
        mask = strcmp(group_col, group_names{i}) & strcmp(model_col, model_names{j});
        T = all_results_table(mask, :);
        if height(T) == 0
            continue;
        end
        T = sortrows(T, {'R2_P', 'RMSEP'}, {'descend', 'ascend'});
        rows(end + 1, :) = table2cell(T(1, :)); %#ok<AGROW>
    end
end
summary_table = cell2table(rows, 'VariableNames', report_var_names());
end

function diagnose_property_targets(project_root, property_name)
% 训练前检查目标列是否存在缺失、无穷值或异常分布。
all_csv_path = fullfile(project_root, 'data', 'physical', 'all_csv_data.csv');
T = readtable(all_csv_path, 'VariableNamingRule', 'preserve');
if ~ismember(property_name, T.Properties.VariableNames)
    error('Target column not found: %s', property_name);
end
y = T{:, property_name};
if iscell(y)
    y = str2double(string(y));
end
y = double(y(:));
valid_y = y(isfinite(y));
if isempty(valid_y)
    error('Target column has no finite values: %s', property_name);
end
fprintf('Target diagnostics | n=%d | mean=%.6f | std=%.6f | min=%.6f | max=%.6f\n', numel(y), mean(valid_y), std(valid_y), min(valid_y), max(valid_y));
end

function cleanup_temp_datasets(temp_dataset_dirs)
unique_dirs = unique(temp_dataset_dirs, 'stable');
for i = 1:numel(unique_dirs)
    d = unique_dirs{i};
    if exist(d, 'dir')
        try
            rmdir(d, 's');
        catch
        end
    end
end
end

function cleanup_temp_runtime(project_root)
paths_to_try = {fullfile(project_root, 'Result', 'Temp'), fullfile(project_root, 'Result', 'Black_White', 'post_processing_data.csv')};
for i = 1:numel(paths_to_try)
    p = paths_to_try{i};
    if exist(p, 'dir')
        try
            rmdir(p, 's');
        catch
        end
    elseif exist(p, 'file')
        try
            delete(p);
        catch
        end
    end
end
end

function txt = param_to_text(param)
if isempty(param)
    txt = 'default';
elseif isnumeric(param)
    txt = mat2str(param);
elseif ischar(param)
    txt = param;
elseif isstring(param)
    txt = char(param);
elseif isstruct(param)
    txt = struct_to_single_line_text(param);
else
    txt = 'custom';
end
txt = sanitize_inline_text(txt);
end

function txt = struct_to_single_line_text(s)
fields = fieldnames(s);
parts = cell(1, numel(fields));
for i = 1:numel(fields)
    name = fields{i};
    value = s.(name);
    parts{i} = sprintf('%s=%s', name, value_to_inline_text(value));
end
txt = strjoin(parts, ', ');
end

function txt = value_to_inline_text(value)
if isnumeric(value) || islogical(value)
    txt = mat2str(value);
elseif ischar(value)
    txt = value;
elseif isstring(value)
    txt = char(value);
elseif iscell(value)
    txt = ['{' strjoin(cellfun(@value_to_inline_text, value, 'UniformOutput', false), ', ') '}'];
elseif isstruct(value)
    txt = ['(' struct_to_single_line_text(value) ')'];
else
    txt = strtrim(evalc('disp(value)'));
end
txt = sanitize_inline_text(txt);
end

function txt = sanitize_inline_text(txt)
txt = char(string(txt));
txt = strrep(txt, sprintf('\r'), ' ');
txt = strrep(txt, sprintf('\n'), ' ');
txt = regexprep(txt, '\s+', ' ');
txt = strtrim(txt);
end

function write_param_details_file(summary_dir, all_results_table, summary_table)
lines = {};
lines{end + 1} = '=== all_results 参数详情 ===';
lines = append_table_param_details(lines, all_results_table);
lines{end + 1} = ' ';
lines{end + 1} = '=== best_models 参数详情 ===';
lines = append_table_param_details(lines, summary_table);
writecell(lines(:), fullfile(summary_dir, 'param_details.txt'), 'FileType', 'text');
end

function lines = append_table_param_details(lines, T)
for i = 1:height(T)
    lines{end + 1} = sprintf('[%d] 组=%s | 模型=%s | 特征=%s | 参数=%s', ...
        i, char(string(T{i, '数据组名称'})), char(string(T{i, '回归器名称'})), ...
        char(string(T{i, '特征筛选参数'})), char(string(T{i, '输入参数'})));
    lines{end + 1} = sprintf('    最优详情: %s', char(string(T{i, '最优参数详情'})));
    lines{end + 1} = sprintf('    指标: R2_C=%.4f | R2_P=%.4f | RMSEC=%.4f | RMSEP=%.4f | RPD=%.4f', ...
        T{i, 'R2_C'}, T{i, 'R2_P'}, T{i, 'RMSEC'}, T{i, 'RMSEP'}, T{i, 'RPD'});
    lines{end + 1} = ' ';
end
end

function print_eta(step_id, step_total, global_tic)
elapsed_sec = toc(global_tic);
avg_sec = elapsed_sec / max(step_id, 1);
remain_sec = avg_sec * (step_total - step_id);
fprintf('Elapsed: %s | ETA: %s\n', format_duration(elapsed_sec), format_duration(remain_sec));
end

function tracker = print_best_update_if_needed(tracker, group_name, fs_method, fs_param_text, model_display_name, train_param_text, result)
is_better = ~tracker.initialized || result.R2_P > tracker.r2p || ...
    (abs(result.R2_P - tracker.r2p) <= 1e-12 && result.RMSEP < tracker.rmsep);
if ~is_better
    return;
end
tracker.initialized = true;
tracker.r2p = result.R2_P;
tracker.rmsep = result.RMSEP;
tracker.summary = sprintf('[Current Best] group=%s | fs=%s%s | model=%s | R2_P=%.4f | RMSEP=%.4f | RPD=%.4f | param=%s', ...
    group_name, upper(fs_method), format_fs_suffix(fs_param_text), model_display_name, result.R2_P, result.RMSEP, result.RPD, train_param_text);
fprintf('%s\n', tracker.summary);
end

function suffix = format_fs_suffix(fs_param_text)
if isempty(fs_param_text)
    suffix = '';
else
    suffix = ['=' fs_param_text];
end
end

function txt = format_duration(sec)
sec = max(0, round(sec));
h = floor(sec / 3600);
m = floor(mod(sec, 3600) / 60);
s = mod(sec, 60);
if h > 0
    txt = sprintf('%02d:%02d:%02d', h, m, s);
else
    txt = sprintf('%02d:%02d', m, s);
end
end

function name = get_model_display_name(method)
switch lower(method)
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

function tag = property_to_tag(property_name)
tag = lower(strtrim(property_name));
tag = strrep(tag, '*', '_star');
tag = regexprep(tag, '[^a-zA-Z0-9_]+', '_');
if isempty(tag)
    tag = 'property';
end
end

function txt = band_idx_to_text(idx)
idx = idx(:)';
if isempty(idx)
    txt = '';
elseif numel(idx) <= 40
    txt = strjoin(arrayfun(@num2str, idx, 'UniformOutput', false), ';');
else
    txt = sprintf('%s;...;%s (total=%d)', strjoin(arrayfun(@num2str, idx(1:20), 'UniformOutput', false), ';'), strjoin(arrayfun(@num2str, idx(end-19:end), 'UniformOutput', false), ';'), numel(idx));
end
end

function opts = default_generalization_options()
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
opts.evaluation = struct('enforce_small_rc_rp_gap', false, 'rc_rp_gap_threshold', 0.05, 'save_smooth_and_feature_plots', true);
end

function opts = normalize_generalization_options(opts)
opts = merge_struct_recursive(default_generalization_options(), opts);
end

function out = merge_struct_recursive(base, override)
out = base;
if ~isstruct(override)
    return;
end
fields = fieldnames(override);
for i = 1:numel(fields)
    key = fields{i};
    if isfield(base, key) && isstruct(base.(key)) && isstruct(override.(key))
        out.(key) = merge_struct_recursive(base.(key), override.(key));
    else
        out.(key) = override.(key);
    end
end
end

function regressor_params = apply_model_complexity_switches(regressor_params, model_opts)
% 根据“模型简化”开关收缩参数空间，优先抑制过拟合风险。
if ~model_opts.simplify_pls_pcr
else
    for i = 1:numel(regressor_params.pls)
        regressor_params.pls{i}.max_lv = min(regressor_params.pls{i}.max_lv, model_opts.pls_max_lv_cap);
        regressor_params.pls{i}.cv_fold = min(regressor_params.pls{i}.cv_fold, 5);
    end
    for i = 1:numel(regressor_params.pcr)
        regressor_params.pcr{i}.max_pc = min(regressor_params.pcr{i}.max_pc, model_opts.pcr_max_pc_cap);
    end
end
if isfield(regressor_params, 'xgboost') && model_opts.xgboost_simplify
    for i = 1:numel(regressor_params.xgboost)
        regressor_params.xgboost{i}.max_num_splits = min(regressor_params.xgboost{i}.max_num_splits, 2^model_opts.xgb_max_depth_cap - 1);
        regressor_params.xgboost{i}.learn_rate = min(regressor_params.xgboost{i}.learn_rate, model_opts.xgb_learn_rate_cap);
        regressor_params.xgboost{i}.min_leaf_size = max(regressor_params.xgboost{i}.min_leaf_size, model_opts.xgb_min_leaf_floor);
        regressor_params.xgboost{i}.num_learning_cycles = min(regressor_params.xgboost{i}.num_learning_cycles, 250);
    end
end
if isfield(regressor_params, 'cnn') && model_opts.cnn_simplify
    for i = 1:numel(regressor_params.cnn)
        regressor_params.cnn{i}.conv_channels = regressor_params.cnn{i}.conv_channels(1:min(numel(regressor_params.cnn{i}.conv_channels), model_opts.cnn_max_blocks));
        regressor_params.cnn{i}.fc_units = min(regressor_params.cnn{i}.fc_units, model_opts.cnn_fc_units_cap);
        regressor_params.cnn{i}.max_epochs = min(regressor_params.cnn{i}.max_epochs, 100);
    end
end
end

function txt = generalization_options_to_text(opts)
parts = {};
if opts.feature.nested_cv_selection, parts{end + 1} = 'nested_fs'; end %#ok<AGROW>
if opts.feature.use_stable_features, parts{end + 1} = 'stable_fs'; end %#ok<AGROW>
if isfield(opts.feature, 'post_feature_pca_projection') && opts.feature.post_feature_pca_projection
    parts{end + 1} = ['post_pca_' num2str(opts.feature.post_feature_pca_components)]; %#ok<AGROW>
end
if opts.data_processing.data_augmentation, parts{end + 1} = 'augment'; end %#ok<AGROW>
if opts.validation.use_external_holdout, parts{end + 1} = 'external'; end %#ok<AGROW>
if opts.model.simplify_pls_pcr, parts{end + 1} = 'simple_pls_pcr'; end %#ok<AGROW>
if opts.model.l1_l2_regularization, parts{end + 1} = 'l2'; end %#ok<AGROW>
if opts.model.dropout, parts{end + 1} = 'dropout'; end %#ok<AGROW>
if opts.model.early_stopping, parts{end + 1} = 'early_stop'; end %#ok<AGROW>
if opts.model.xgboost_simplify, parts{end + 1} = 'xgb_simple'; end %#ok<AGROW>
if opts.model.cnn_simplify, parts{end + 1} = 'cnn_simple'; end %#ok<AGROW>
parts{end + 1} = ['xgb_grid_' char(string(opts.model.xgboost_grid_mode))]; %#ok<AGROW>
parts{end + 1} = ['cnn_grid_' char(string(opts.model.cnn_grid_mode))]; %#ok<AGROW>
if isempty(parts)
    txt = '全部关闭';
else
    txt = strjoin(parts, ' | ');
end
end

function grid = build_xgboost_param_grid(model_opts)
% XGBoost 专用参数网格：
% off=单组参数，quick=小网格，full=系统搜索。
custom_grid = build_custom_xgboost_grid(model_opts);
if ~isempty(custom_grid)
    grid = custom_grid;
    return;
end
mode = lower(strtrim(char(model_opts.xgboost_grid_mode)));
switch mode
    case 'off'
        grid = {struct('num_learning_cycles', 300, 'learn_rate', 0.05, 'max_num_splits', 15, 'min_leaf_size', 5)};
    case 'quick'
        grid = { ...
            struct('num_learning_cycles', 200, 'learn_rate', 0.05, 'max_num_splits', 15, 'min_leaf_size', 5), ...
            struct('num_learning_cycles', 300, 'learn_rate', 0.03, 'max_num_splits', 15, 'min_leaf_size', 8), ...
            struct('num_learning_cycles', 400, 'learn_rate', 0.03, 'max_num_splits', 31, 'min_leaf_size', 8)};
    case 'full'
        cycles = [150 250 400 600];
        rates = [0.01 0.03 0.05];
        splits = [7 15 31];
        leaves = [3 5 8 12];
        idx = 0;
        grid = cell(numel(cycles) * numel(rates) * numel(splits) * numel(leaves), 1);
        for i = 1:numel(cycles)
            for j = 1:numel(rates)
                for k = 1:numel(splits)
                    for t = 1:numel(leaves)
                        idx = idx + 1;
                        grid{idx} = struct('num_learning_cycles', cycles(i), 'learn_rate', rates(j), 'max_num_splits', splits(k), 'min_leaf_size', leaves(t));
                    end
                end
            end
        end
    otherwise
        error('Unsupported xgboost_grid_mode: %s', mode);
end
end

function grid = build_custom_xgboost_grid(model_opts)
cycles = numeric_row_or_empty(model_opts, 'xgb_num_learning_cycles');
rates = numeric_row_or_empty(model_opts, 'xgb_learn_rates');
splits = numeric_row_or_empty(model_opts, 'xgb_max_num_splits');
leaves = numeric_row_or_empty(model_opts, 'xgb_min_leaf_sizes');
if isempty(cycles) || isempty(rates) || isempty(splits) || isempty(leaves)
    grid = {};
    return;
end
idx = 0;
grid = cell(numel(cycles) * numel(rates) * numel(splits) * numel(leaves), 1);
for i = 1:numel(cycles)
    for j = 1:numel(rates)
        for k = 1:numel(splits)
            for t = 1:numel(leaves)
                idx = idx + 1;
                grid{idx} = struct('num_learning_cycles', cycles(i), 'learn_rate', rates(j), 'max_num_splits', splits(k), 'min_leaf_size', leaves(t));
            end
        end
    end
end
end

function grid = build_cnn_param_grid(model_opts)
% CNN 专用参数网格：
% off=单组参数，quick=小网格，full=系统搜索。
custom_grid = build_custom_cnn_grid(model_opts);
if ~isempty(custom_grid)
    grid = custom_grid;
    return;
end
mode = lower(strtrim(char(model_opts.cnn_grid_mode)));
switch mode
    case 'off'
        grid = {struct('conv_channels', [16 32], 'fc_units', 64, 'dropout_rate', 0.20, 'max_epochs', 100, 'mini_batch_size', 16, 'initial_learn_rate', 1e-3)};
    case 'quick'
        grid = { ...
            struct('conv_channels', [16 32], 'fc_units', 48, 'dropout_rate', 0.20, 'max_epochs', 100, 'mini_batch_size', 16, 'initial_learn_rate', 1e-3), ...
            struct('conv_channels', [16 32], 'fc_units', 64, 'dropout_rate', 0.30, 'max_epochs', 120, 'mini_batch_size', 16, 'initial_learn_rate', 5e-4), ...
            struct('conv_channels', [32 64], 'fc_units', 64, 'dropout_rate', 0.30, 'max_epochs', 120, 'mini_batch_size', 8, 'initial_learn_rate', 5e-4)};
    case 'full'
        conv_sets = {[8 16], [16 32], [16 32 64], [32 64]};
        fc_units = [32 48 64 96];
        dropout_rates = [0.10 0.20 0.30 0.40];
        max_epochs = [80 120 160];
        batch_sizes = [8 16];
        learn_rates = [1e-3 5e-4];
        idx = 0;
        grid = cell(numel(conv_sets) * numel(fc_units) * numel(dropout_rates) * numel(max_epochs) * numel(batch_sizes) * numel(learn_rates), 1);
        for a = 1:numel(conv_sets)
            for b = 1:numel(fc_units)
                for c = 1:numel(dropout_rates)
                    for d = 1:numel(max_epochs)
                        for e = 1:numel(batch_sizes)
                            for f = 1:numel(learn_rates)
                                idx = idx + 1;
                                grid{idx} = struct('conv_channels', conv_sets{a}, 'fc_units', fc_units(b), 'dropout_rate', dropout_rates(c), 'max_epochs', max_epochs(d), 'mini_batch_size', batch_sizes(e), 'initial_learn_rate', learn_rates(f));
                            end
                        end
                    end
                end
            end
        end
    otherwise
        error('Unsupported cnn_grid_mode: %s', mode);
end
end

function grid = build_custom_cnn_grid(model_opts)
conv_sets = cell_grid_or_empty(model_opts, 'cnn_conv_channel_sets');
fc_units = numeric_row_or_empty(model_opts, 'cnn_fc_units_grid');
dropout_rates = numeric_row_or_empty(model_opts, 'cnn_dropout_rates');
max_epochs = numeric_row_or_empty(model_opts, 'cnn_max_epochs_grid');
batch_sizes = numeric_row_or_empty(model_opts, 'cnn_mini_batch_sizes');
learn_rates = numeric_row_or_empty(model_opts, 'cnn_initial_learn_rates');
if isempty(conv_sets) || isempty(fc_units) || isempty(dropout_rates) || isempty(max_epochs) || isempty(batch_sizes) || isempty(learn_rates)
    grid = {};
    return;
end
idx = 0;
grid = cell(numel(conv_sets) * numel(fc_units) * numel(dropout_rates) * numel(max_epochs) * numel(batch_sizes) * numel(learn_rates), 1);
for a = 1:numel(conv_sets)
    for b = 1:numel(fc_units)
        for c = 1:numel(dropout_rates)
            for d = 1:numel(max_epochs)
                for e = 1:numel(batch_sizes)
                    for f = 1:numel(learn_rates)
                        idx = idx + 1;
                        grid{idx} = struct('conv_channels', conv_sets{a}, 'fc_units', fc_units(b), 'dropout_rate', dropout_rates(c), 'max_epochs', max_epochs(d), 'mini_batch_size', batch_sizes(e), 'initial_learn_rate', learn_rates(f));
                    end
                end
            end
        end
    end
end
end

function values = numeric_row_or_empty(s, field_name)
if ~isfield(s, field_name) || isempty(s.(field_name))
    values = [];
    return;
end
values = unique(double(s.(field_name)(:)'));
end

function values = cell_grid_or_empty(s, field_name)
if ~isfield(s, field_name) || isempty(s.(field_name))
    values = {};
    return;
end
raw = s.(field_name);
if isnumeric(raw)
    values = {raw};
elseif iscell(raw)
    values = raw;
else
    values = {raw};
end
end

function out = ternary_text(cond, true_text, false_text)
if cond
    out = true_text;
else
    out = false_text;
end
end
