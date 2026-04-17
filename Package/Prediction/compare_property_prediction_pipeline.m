function summary = compare_property_prediction_pipeline(property_name, filter_method, feature_selection_method, sg_order, sg_window, include_preprocessed_group)
% 功能：比较两组数据在不同回归器下的预测效果
% 第 1 组：仅做滤波/预处理
% 第 2 组：先滤波/预处理，再进行特征筛选
% 用法示例：
%   summary = compare_property_prediction_pipeline('a*')
%   summary = compare_property_prediction_pipeline('a*', 'sg+msc+snv', 'cars')
%   summary = compare_property_prediction_pipeline('L*', 'sg+msc+snv', 'spa', 2, 15)
%   summary = compare_property_prediction_pipeline('L*', 'sg+msc+snv', 'spa', 2, 15, false)

if nargin < 1 || isempty(property_name)
    error('property_name 必填，例如：''a*'' 或 ''L*''。');
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
% false代表不选只筛选数据
if nargin < 6 || isempty(include_preprocessed_group)
    include_preprocessed_group = false;
end

if iscell(feature_selection_method) || (isstring(feature_selection_method) && numel(feature_selection_method) > 1)
    method_list = cellstr(feature_selection_method);
    summary_parts = cell(numel(method_list), 1);
    for ii = 1:numel(method_list)
        summary_parts{ii} = compare_property_prediction_pipeline(property_name, filter_method, method_list{ii}, sg_order, sg_window, include_preprocessed_group);
    end
    try
        summary = vertcat(summary_parts{:});
    catch
        summary = summary_parts;
    end
    return;
end

clc;
close all;

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(project_root));

% ==================== 参数区（在这里统一修改） ====================
preproc_mode = lower(strtrim(filter_method));
fs_method = lower(strtrim(feature_selection_method));

sg_order = sg_order;         % SG 多项式阶数
sg_window = sg_window;       % SG 窗口长度
msc_ref_mode = 'mean';       % MSC 参考方式：mean / median / first
snv_mode = 'robust';          % SNV 模式：standard / robust
keep_dataset_exports = false; % false 表示过程数据只保留临时 dataset.mat
baseline_zero_mode = 'first_5_mean'; % 可选：'none' / 'first_point' / 'first_5_mean' %基线处理
baseline_zero_scope = 'full_spectrum'; % 可选：'cropped_spectrum' / 'full_spectrum' %基线归零作用范围
despike_mode = 'jump_guard';   % 可选：'none' / 'median3' / 'median5' / 'median7' / 'local' / 'local_strong' / 'jump_guard' %去尖刺

% 回归器搜索模式：
gpr_mode = 'fast';   % 可选：'fast' / 'full'

feature_selection_param_grid = struct();
small_feature_mode = false;   % true 时才把特征数压到 2~5；近红外默认建议 false
if small_feature_mode
    feature_selection_param_grid.cars = {struct('num_sampling_runs', 20, 'target_count', 2), struct('num_sampling_runs', 20, 'target_count', 3), struct('num_sampling_runs', 20, 'target_count', 4), struct('num_sampling_runs', 20, 'target_count', 5)};
    feature_selection_param_grid.pca = {2, 3, 4, 5};
    feature_selection_param_grid.corr_topk = {2, 3, 4, 5};
    feature_selection_param_grid.spa = {2, 3, 4, 5};
else
    feature_selection_param_grid.cars = {120,150}; % 20, 30, 40, 80
    feature_selection_param_grid.pca = {40, 80, 120, 160, 200};
    feature_selection_param_grid.corr_topk = {20, 40, 80, 120};
    feature_selection_param_grid.spa = {8, 12, 16, 20};
end

% regressors = {'pls', 'pcr', 'svr', 'rf', 'gpr', 'knn'}; 
regressors = {'pls', 'pcr'}; % RF / GPR / KNN 参数网格也可在这里直接修改
regressor_params = struct();
% 旧默认 PLS 参数：max_lv=256, cv_fold=10
regressor_params.pls = {struct('max_lv', 300, 'cv_fold', 10)};
% 旧默认 PCR 参数：max_pc=200
regressor_params.pcr = {struct('max_pc', 300)};
% 旧默认 SVR 参数：'fast' -> kernels={'linear','gaussian'}, boxes=[1 10], scales={'auto'}
regressor_params.svr = {struct('kernels', {{'linear', 'gaussian'}}, 'boxes', [0.1 0.5 1 5], 'scales', {{'auto', 0.5, 1, 2, 5, 10}})};
% 旧默认 RF 参数：num_trees=[100 200], min_leaf=[1 5 10]
regressor_params.rf = {struct( 'num_trees', [100 200], 'min_leaf', [1 5 10 20 30])};
% GPR 参数模式：fast 更快，full 更全
switch lower(strtrim(gpr_mode))
    case 'fast'
        regressor_params.gpr = {struct('kernels', {{'squaredexponential', 'matern32', 'matern52'}})};
    case 'full'
        regressor_params.gpr = {struct('kernels', {{'squaredexponential', 'ardsquaredexponential', 'exponential', 'ardexponential', 'matern32', 'matern52', 'ardmatern32', 'ardmatern52', 'rationalquadratic', 'ardrationalquadratic'}})};
    otherwise
        error('不支持的 gpr_mode：%s。可选：fast / full', gpr_mode);
end
% 旧默认 KNN 参数：'fast' -> k=[3 5 7], distance={'euclidean'}, weighting={'uniform','inverse'}
regressor_params.knn = {struct('k', [1 3 5 7 9 11 15 21 31], 'distances', {{'euclidean', 'cityblock', 'chebychev', 'cosine', 'correlation'}}, 'weightings', {{'uniform', 'inverse'}})};
% ================================================================

if ~isfield(feature_selection_param_grid, fs_method)
    error('不支持的特征选择方式：%s', fs_method);
end
fs_param_grid = feature_selection_param_grid.(fs_method);

run_tag = datestr(now, 'yyyymmdd_HHMMSS');
setenv('HXR_RUN_TAG', run_tag);

property_tag = property_to_tag(property_name);
summary_dir = fullfile(project_root, 'Result', 'Summary', [property_tag '_regressor_compare_' run_tag]);
if ~exist(summary_dir, 'dir')
    mkdir(summary_dir);
end

step_total = 0;
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

step_id = 0;
all_rows = {};
temp_dataset_dirs = {};
global_tic = tic;

% 在任何数据处理前，先检查目标列是否存在异常值
diagnose_property_targets(project_root, property_name);

fprintf('================ 回归器对比开始 ================\n');
fprintf('预测对象：%s\n', property_name);
if include_preprocessed_group
    fprintf('对比组 1：仅滤波（%s）\n', upper(preproc_mode));
else
    fprintf('对比组 1：已关闭\n');
end
fprintf('对比组 2：滤波 + 特征筛选（%s）\n', upper(fs_method));
fprintf('SG 阶数=%d | SG 窗口=%d\n', sg_order, sg_window);
fprintf('MSC 模式=%s | SNV 模式=%s | 基线归零=%s | 归零范围=%s | 去尖刺=%s\n', msc_ref_mode, snv_mode, baseline_zero_mode, baseline_zero_scope, despike_mode);
fprintf('特征筛选参数候选：%s\n', strjoin(cellfun(@param_to_text, fs_param_grid, 'UniformOutput', false), ' / '));
fprintf('回归器：%s\n', strjoin(cellfun(@get_model_display_name, regressors, 'UniformOutput', false), ' / '));
fprintf('结果目录：%s\n', summary_dir);
fprintf('总组合数：%d\n', step_total);

if include_preprocessed_group
    fprintf('\n[阶段 1/2] 准备仅滤波数据集...\n');
    dataset_pre = prepare_property_dataset(property_name, 'preprocessed', preproc_mode, sg_order, sg_window, 'corr_topk', [], msc_ref_mode, snv_mode, keep_dataset_exports, baseline_zero_mode, despike_mode, baseline_zero_scope);
    temp_dataset_dirs{end + 1} = dataset_pre.paths.dir;
    for i = 1:numel(regressors)
        method_name = regressors{i};
        params = regressor_params.(method_name);
        for j = 1:numel(params)
            step_id = step_id + 1;
            method_param = params{j};
            fprintf('\n[进度 %d/%d] 仅滤波 | 回归器=%s | 参数=%s\n', step_id, step_total, get_model_display_name(method_name), param_to_text(method_param));
            result = train_model_from_dataset(dataset_pre.paths.mat, method_name, method_param);
            print_eta(step_id, step_total, global_tic);
            all_rows(end + 1, :) = build_report_row('仅滤波', 'none', '', get_model_display_name(method_name), param_to_text(method_param), result); %#ok<AGROW>
        end
    end
else
    fprintf('\n[阶段 1/2] 已跳过仅滤波组训练。\n');
end

fprintf('\n[阶段 2/2] 准备特征筛选后数据集...\n');
for i = 1:numel(fs_param_grid)
    fs_param = fs_param_grid{i};
    fprintf('\n---- 当前特征筛选：%s | 参数=%s ----\n', upper(fs_method), param_to_text(fs_param));
    dataset_sel = prepare_property_dataset(property_name, 'selected', preproc_mode, sg_order, sg_window, fs_method, fs_param, msc_ref_mode, snv_mode, keep_dataset_exports, baseline_zero_mode, despike_mode, baseline_zero_scope);
    temp_dataset_dirs{end + 1} = dataset_sel.paths.dir;
    for j = 1:numel(regressors)
        method_name = regressors{j};
        params = regressor_params.(method_name);
        for k = 1:numel(params)
            step_id = step_id + 1;
            method_param = params{k};
            fprintf('\n[进度 %d/%d] 特征筛选后 | %s=%s | 回归器=%s | 参数=%s\n', ...
                step_id, step_total, upper(fs_method), param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param));
            result = train_model_from_dataset(dataset_sel.paths.mat, method_name, method_param);
            print_eta(step_id, step_total, global_tic);
            all_rows(end + 1, :) = build_report_row('特征筛选后', fs_method, param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param), result); %#ok<AGROW>
        end
    end
end
all_results_table = cell2table(all_rows, 'VariableNames', report_var_names());
all_results_table = sortrows(all_results_table, {'数据组名称', '回归器名称', 'R2_P', 'RMSEP'}, {'ascend', 'ascend', 'descend', 'ascend'});
summary_table = build_best_group_model_table(all_results_table);
save(fullfile(summary_dir, 'summary.mat'), 'all_results_table', 'summary_table');
writetable(all_results_table, fullfile(summary_dir, 'all_results.csv'));

cleanup_temp_datasets(temp_dataset_dirs);
cleanup_temp_runtime(project_root);
setenv('HXR_RUN_TAG', '');

summary = all_results_table;

fprintf('\n================ 回归器对比完成 ================\n');
fprintf('总耗时：%s\n', format_duration(toc(global_tic)));
disp(all_results_table);
fprintf('结果已保存到：%s\n', summary_dir);
fprintf('临时数据已清理。\n');
end

function diagnose_property_targets(project_root, property_name)
all_csv_path = fullfile(project_root, 'data', 'physical', 'all_csv_data.csv');
T = readtable(all_csv_path, 'VariableNamingRule', 'preserve');
if ~ismember(property_name, T.Properties.VariableNames)
    error('all_csv_data.csv 中不存在目标列：%s', property_name);
end

y = T{:, property_name};
if iscell(y)
    y = str2double(string(y));
end
y = double(y(:));
valid = isfinite(y);
valid_y = y(valid);
bad_idx = find(~valid);

has_csv_name = ismember('csv_name', T.Properties.VariableNames);
if has_csv_name
    csv_names = string(T{:, 'csv_name'});
else
    csv_names = strings(height(T), 1);
end

fprintf('\n================ 目标值诊断 ================\n');
fprintf('预测对象：%s\n', property_name);
fprintf('原始样本数：%d\n', numel(y));
fprintf('有效值数量：%d\n', nnz(valid));
fprintf('NaN/Inf 数量：%d\n', nnz(~valid));

if ~isempty(bad_idx)
    fprintf('异常行（行号 -> csv_name）：\n');
    show_n = min(20, numel(bad_idx));
    for i = 1:show_n
        idx = bad_idx(i);
        if has_csv_name
            fprintf('  %d -> %s\n', idx, csv_names(idx));
        else
            fprintf('  %d\n', idx);
        end
    end
    if numel(bad_idx) > show_n
        fprintf('  ... 其余还有 %d 行异常\n', numel(bad_idx) - show_n);
    end

    report = table(bad_idx, y(bad_idx), 'VariableNames', {'row_index', 'raw_value'});
    if has_csv_name
        report.csv_name = csv_names(bad_idx);
    end
    report_path = fullfile(project_root, 'Result', 'Summary', ['diag_' property_to_tag(property_name) '_invalid_rows.csv']);
    try
        writetable(report, report_path);
        fprintf('异常行报告已保存：%s\n', report_path);
    catch
    end
end

if isempty(valid_y)
    error('目标列 %s 没有有效数值。', property_name);
end

fprintf('均值：%.6f\n', mean(valid_y));
fprintf('标准差：%.6f\n', std(valid_y));
fprintf('最小值：%.6f\n', min(valid_y));
fprintf('最大值：%.6f\n', max(valid_y));
fprintf('唯一值个数：%d\n', numel(unique(valid_y)));
q = quantile(valid_y, [0.25 0.50 0.75]);
fprintf('Q1：%.6f | 中位数：%.6f | Q3：%.6f\n', q(1), q(2), q(3));
fprintf('===========================================\n\n');
end

function row = build_report_row(group_name, fs_method, fs_param_text, model_display_name, train_param_text, result)
meta = result.dataset_metadata;
model_mat_path = '';
regression_plot_path = '';
used_band_range_text = '';
used_band_idx_text = '';
if isfield(result, 'result_mat_path')
    model_mat_path = result.result_mat_path;
end
if isfield(result, 'regression_plot_path')
    regression_plot_path = result.regression_plot_path;
end
if isfield(meta, 'used_band_range') && numel(meta.used_band_range) >= 2
    used_band_range_text = sprintf('%d:%d', meta.used_band_range(1), meta.used_band_range(end));
end
if isfield(meta, 'used_band_idx') && ~isempty(meta.used_band_idx)
    used_band_idx_text = band_idx_to_text(meta.used_band_idx);
end
row = { ...
    group_name, ...
    fs_method, ...
    fs_param_text, ...
    model_display_name, ...
    train_param_text, ...
    result.best_param_detail, ...
    meta.preproc_mode, ...
    meta.sg_order, ...
    meta.sg_window, ...
    meta.msc_ref_mode, ...
    meta.snv_mode, ...
    meta.baseline_zero_mode, ...
    meta.despike_mode, ...
    used_band_range_text, ...
    used_band_idx_text, ...
    result.R2_C, ...
    result.R2_P, ...
    result.RMSEC, ...
    result.RMSEP, ...
    result.RPD, ...
    model_mat_path, ...
    regression_plot_path};
end

function cleanup_temp_datasets(temp_dataset_dirs)
if isempty(temp_dataset_dirs)
    return;
end
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
paths_to_try = { ...
    fullfile(project_root, 'Result', 'Temp'), ...
    fullfile(project_root, 'Result', 'Black_White', 'post_processing_data.csv')};
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

function names = report_var_names()
names = {'数据组名称', '特征筛选方法', '特征筛选参数', '回归器名称', '输入参数', '最优参数详情', ...
    '预处理方式', 'SG阶数', 'SG窗口', 'MSC模式', 'SNV模式', '基线归零模式', '去尖刺模式', ...
    '使用波段范围', '使用波段索引', ...
    'R2_C', 'R2_P', 'RMSEC', 'RMSEP', 'RPD', ...
    '模型结果MAT路径', '回归图像路径'};
end

function summary_table = build_best_group_model_table(all_results_table)
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
summary_table = sortrows(summary_table, {'数据组名称', 'R2_P', 'RMSEP'}, {'ascend', 'descend', 'ascend'});
end

function report_top5 = build_top5_report(all_results_table)
rows = {};
model_col = all_results_table{:, '回归器名称'};
model_names = unique(model_col, 'stable');
for i = 1:numel(model_names)
    model_display_name = model_names{i};
    T = all_results_table(strcmp(model_col, model_display_name), :);
    T = sortrows(T, {'R2_P', 'RMSEP'}, {'descend', 'ascend'});
    top_n = min(5, height(T));
    for k = 1:top_n
        rows(end + 1, :) = [{k}, table2cell(T(k, :))]; %#ok<AGROW>
    end
end
report_top5 = cell2table(rows, 'VariableNames', [{'排名'}, report_var_names()]);
end

function txt = param_to_text(param)
if isempty(param)
    txt = '默认';
elseif isnumeric(param)
    txt = mat2str(param);
elseif ischar(param)
    txt = param;
elseif iscell(param)
    txt = 'cell';
else
    txt = 'custom';
end
end

function print_eta(step_id, step_total, global_tic)
elapsed_sec = toc(global_tic);
avg_sec = elapsed_sec / max(step_id, 1);
remain_steps = step_total - step_id;
remain_sec = avg_sec * remain_steps;
fprintf('已耗时：%s | 预计剩余：%s\n', format_duration(elapsed_sec), format_duration(remain_sec));
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
    return;
end
if numel(idx) <= 40
    txt = strjoin(arrayfun(@num2str, idx, 'UniformOutput', false), ';');
else
    head_txt = strjoin(arrayfun(@num2str, idx(1:20), 'UniformOutput', false), ';');
    tail_txt = strjoin(arrayfun(@num2str, idx(end-19:end), 'UniformOutput', false), ';');
    txt = sprintf('%s;...;%s (共%d个)', head_txt, tail_txt, numel(idx));
end

end
