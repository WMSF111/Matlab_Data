function summary = compare_property_prediction_pipeline(property_name, filter_method, feature_selection_method)
% 功能：比较两组数据在不同回归器下的预测效果
% 第 1 组：仅做滤波/预处理
% 第 2 组：先滤波/预处理，再进行特征筛选
% 用法示例：
%   summary = compare_property_prediction_pipeline('a*')
%   summary = compare_property_prediction_pipeline('a*', 'sg+msc+snv', 'cars')

if nargin < 1 || isempty(property_name)
    error('property_name 必填，例如：''a*'' 或 ''L*''。');
end
if nargin < 2 || isempty(filter_method)
    filter_method = 'sg+msc+snv';
end
if nargin < 3 || isempty(feature_selection_method)
    feature_selection_method = 'pca';
end

clc;
close all;

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

% ==================== 参数区（在这里统一修改） ====================
preproc_mode = lower(strtrim(filter_method));
fs_method = lower(strtrim(feature_selection_method));

sg_order = 3;                 % SG 多项式阶数
sg_window = 35;               % SG 窗口长度
msc_ref_mode = 'first';       % MSC 参考方式：mean / median / first
snv_mode = 'robust';          % SNV 模式：standard / robust
keep_dataset_exports = false; % false 表示过程数据只保留临时 dataset.mat

feature_selection_param_grid = struct();
feature_selection_param_grid.cars = {20, 30, 40, 80};
feature_selection_param_grid.pca = {20, 80, 160, 200};
feature_selection_param_grid.corr_topk = {20, 40, 80};
feature_selection_param_grid.spa = {8, 12};

regressors = {'pls', 'pcr', 'svr', 'rf', 'gpr', 'knn'};
regressor_params = struct();
regressor_params.pls = {[]};
regressor_params.pcr = {200};
regressor_params.svr = {'fast'};
regressor_params.rf = {[]};
regressor_params.gpr = {'fast'};
regressor_params.knn = {'fast'};
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
for i = 1:numel(regressors)
    step_total = step_total + numel(regressor_params.(regressors{i}));
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
fprintf('对比组 1：仅滤波（%s）\n', upper(preproc_mode));
fprintf('对比组 2：滤波 + 特征筛选（%s）\n', upper(fs_method));
fprintf('SG 阶数=%d | SG 窗口=%d\n', sg_order, sg_window);
fprintf('MSC 模式=%s | SNV 模式=%s\n', msc_ref_mode, snv_mode);
fprintf('特征筛选参数候选：%s\n', strjoin(cellfun(@param_to_text, fs_param_grid, 'UniformOutput', false), ' / '));
fprintf('回归器：%s\n', strjoin(cellfun(@get_model_display_name, regressors, 'UniformOutput', false), ' / '));
fprintf('结果目录：%s\n', summary_dir);
fprintf('总组合数：%d\n', step_total);

fprintf('\n[阶段 1/2] 准备仅滤波数据集...\n');
dataset_pre = prepare_property_dataset(property_name, 'preprocessed', preproc_mode, sg_order, sg_window, 'corr_topk', [], msc_ref_mode, snv_mode, keep_dataset_exports);
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

fprintf('\n[阶段 2/2] 准备特征筛选后数据集...\n');
for i = 1:numel(fs_param_grid)
    fs_param = fs_param_grid{i};
    fprintf('\n---- 当前特征筛选：%s | 参数=%s ----\n', upper(fs_method), param_to_text(fs_param));
    dataset_sel = prepare_property_dataset(property_name, 'selected', preproc_mode, sg_order, sg_window, fs_method, fs_param, msc_ref_mode, snv_mode, keep_dataset_exports);
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
if isfield(result, 'result_mat_path')
    model_mat_path = result.result_mat_path;
end
if isfield(result, 'regression_plot_path')
    regression_plot_path = result.regression_plot_path;
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
    '预处理方式', 'SG阶数', 'SG窗口', 'MSC模式', 'SNV模式', ...
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
