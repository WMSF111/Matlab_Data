function summary = compare_a_prediction_pipeline(msc_ref_mode, snv_mode)
% 功能：
%   1. 以 a* 为预测对象进行回归比较
%   2. 对比组 1：仅做 SG+MSC+SNV 预处理
%   3. 对比组 2：SG+MSC+SNV 后再做 CARS 特征筛选
%   4. 比较回归器：PLS / PCR / SVR / RF / GPR / KNN
%   5. 完整运行后统一保存结果报告

if nargin < 1 || isempty(msc_ref_mode), msc_ref_mode = 'mean'; end
if nargin < 2 || isempty(snv_mode), snv_mode = 'standard'; end

clc;
close all;

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(project_root));

property_name = 'a*';
preproc_mode = 'sg+msc+snv';
sg_order = 3;
sg_window = 35;
keep_dataset_exports = false;
run_tag = datestr(now, 'yyyymmdd_HHMMSS');
setenv('HXR_RUN_TAG', run_tag);

regressors = {'pls', 'pcr', 'svr', 'rf', 'gpr', 'knn'};
regressor_params = struct();
regressor_params.pls = {[]};
regressor_params.pcr = {200};
regressor_params.svr = {'fast'};
regressor_params.rf = {[]};
regressor_params.gpr = {'fast'};
regressor_params.knn = {'fast'};

cars_fs_params = {20, 30, 40};

summary_dir = fullfile(project_root, 'Result', 'Summary', ['a_star_regressor_compare_', run_tag]);
if ~exist(summary_dir, 'dir')
    mkdir(summary_dir);
end

temp_dataset_dirs = {};
all_rows = {};
step_id = 0;
step_total = 0;
for j = 1:numel(regressors)
    step_total = step_total + numel(regressor_params.(regressors{j}));
end
for i = 1:numel(cars_fs_params)
    for j = 1:numel(regressors)
        step_total = step_total + numel(regressor_params.(regressors{j}));
    end
end

global_tic = tic;

fprintf('================ a* 回归器比较开始 ================\n');
fprintf('预测对象：%s\n', property_name);
fprintf('对比组 1：SG+MSC+SNV\n');
fprintf('对比组 2：SG+MSC+SNV + CARS 特征筛选\n');
fprintf('SG 参数：阶数=%d，窗口=%d\n', sg_order, sg_window);
fprintf('MSC 模式：%s | SNV 模式：%s\n', msc_ref_mode, snv_mode);
fprintf('回归器：%s\n', strjoin(cellfun(@get_model_display_name, regressors, 'UniformOutput', false), ' / '));
fprintf('CARS 参数候选：%s\n', strjoin(cellfun(@param_to_text, cars_fs_params, 'UniformOutput', false), ' / '));
fprintf('数据集导出模式：仅保留临时 dataset.mat\n');
fprintf('结果目录：%s\n', summary_dir);
fprintf('总组合数：%d\n', step_total);

fprintf('\n[阶段 1/2] 准备第 1 组数据：SG+MSC+SNV\n');
dataset_pre = prepare_property_dataset(property_name, 'preprocessed', preproc_mode, sg_order, sg_window, 'corr_topk', [], msc_ref_mode, snv_mode, keep_dataset_exports);
temp_dataset_dirs{end + 1} = dataset_pre.paths.dir;
for i = 1:numel(regressors)
    method_name = regressors{i};
    params = regressor_params.(method_name);
    for j = 1:numel(params)
        step_id = step_id + 1;
        method_param = params{j};
        fprintf('\n[进度 %d/%d] 第 1 组 | 回归器=%s | 输入参数=%s\n', step_id, step_total, get_model_display_name(method_name), param_to_text(method_param));
        result = train_model_from_dataset(dataset_pre.paths.mat, method_name, method_param);
        print_eta(step_id, step_total, global_tic);
        all_rows(end + 1, :) = build_report_row('SG+MSC+SNV', '无', '', get_model_display_name(method_name), param_to_text(method_param), result); %#ok<AGROW>
    end
end

fprintf('\n[阶段 2/2] 准备第 2 组数据：SG+MSC+SNV + CARS 特征筛选\n');
for i = 1:numel(cars_fs_params)
    fs_param = cars_fs_params{i};
    fprintf('\n---- 第 2 组当前 CARS 参数：%s ----\n', param_to_text(fs_param));
    dataset_sel = prepare_property_dataset(property_name, 'selected', preproc_mode, sg_order, sg_window, 'cars', fs_param, msc_ref_mode, snv_mode, keep_dataset_exports);
    temp_dataset_dirs{end + 1} = dataset_sel.paths.dir;
    for j = 1:numel(regressors)
        method_name = regressors{j};
        params = regressor_params.(method_name);
        for k = 1:numel(params)
            step_id = step_id + 1;
            method_param = params{k};
            fprintf('\n[进度 %d/%d] 第 2 组 | CARS=%s | 回归器=%s | 输入参数=%s\n', ...
                step_id, step_total, param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param));
            result = train_model_from_dataset(dataset_sel.paths.mat, method_name, method_param);
            print_eta(step_id, step_total, global_tic);
            all_rows(end + 1, :) = build_report_row('SG+MSC+SNV + CARS 特征筛选', 'cars', param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param), result); %#ok<AGROW>
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
summary = all_results_table;
setenv('HXR_RUN_TAG', '');

fprintf('\n================ a* 回归器比较完成 ================\n');
fprintf('耗时：%s\n', format_duration(toc(global_tic)));
disp(all_results_table);
fprintf('结果已保存到：%s\n', summary_dir);
fprintf('临时数据已清理。\n');
end
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
