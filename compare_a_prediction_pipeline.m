function summary = compare_a_prediction_pipeline(msc_ref_mode, snv_mode)
% 功能：
%   1) 对 a* 做两组数据方案的回归器对比
%   2) 数据组1：SG+MSC+SNV
%   3) 数据组2：SG+MSC+SNV + CARS筛选
%   4) 比较回归器：PLS / PCR / SVR / RF
%   5) 完整跑完后统一保存报告

if nargin < 1 || isempty(msc_ref_mode), msc_ref_mode = 'mean'; end
if nargin < 2 || isempty(snv_mode), snv_mode = 'standard'; end

clc;
close all;

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

property_name = 'a*';
preproc_mode = 'sg+msc+snv';
sg_order = 3;
sg_window = 35;

regressors = {'pls', 'pcr', 'svr', 'rf'};
regressor_params = struct();
regressor_params.pls = {[]};
regressor_params.pcr = {200};
regressor_params.svr = {'fast'};
regressor_params.rf = {[]};

cars_fs_params = {20, 30, 40};

summary_dir = fullfile(project_root, 'Result', 'Summary', 'a_star_regressor_compare');
if ~exist(summary_dir, 'dir')
    mkdir(summary_dir);
end

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

fprintf('================ 两组数据回归器对比开始 ================\n');
fprintf('目标属性: %s\n', property_name);
fprintf('分组1: SG+MSC+SNV\n');
fprintf('分组2: SG+MSC+SNV + CARS筛选\n');
fprintf('SG参数: 阶数=%d, 窗口=%d\n', sg_order, sg_window);
fprintf('MSC 模式: %s | SNV 模式: %s\n', msc_ref_mode, snv_mode);
fprintf('回归器: %s\n', strjoin(cellfun(@get_model_display_name, regressors, 'UniformOutput', false), ' / '));
fprintf('CARS筛选参数候选: %s\n', strjoin(cellfun(@param_to_text, cars_fs_params, 'UniformOutput', false), ' / '));
fprintf('预计训练组合数: %d\n', step_total);

fprintf('\n[阶段 1/2] 准备分组1数据：SG+MSC+SNV\n');
dataset_pre = prepare_property_dataset(property_name, 'preprocessed', preproc_mode, sg_order, sg_window, 'corr_topk', [], msc_ref_mode, snv_mode);
for i = 1:numel(regressors)
    method_name = regressors{i};
    params = regressor_params.(method_name);
    for j = 1:numel(params)
        step_id = step_id + 1;
        method_param = params{j};
        fprintf('\n[进度 %d/%d] 分组1 | 模型=%s | 参数=%s\n', step_id, step_total, get_model_display_name(method_name), param_to_text(method_param));
        result = train_model_from_dataset(dataset_pre.paths.mat, method_name, method_param);
        print_eta(step_id, step_total, global_tic);
        all_rows(end + 1, :) = build_report_row('SG+MSC+SNV', '无', '', get_model_display_name(method_name), param_to_text(method_param), result); %#ok<AGROW>
    end
end

fprintf('\n[阶段 2/2] 准备分组2数据：SG+MSC+SNV + CARS筛选\n');
for i = 1:numel(cars_fs_params)
    fs_param = cars_fs_params{i};
    fprintf('\n---- 分组2当前CARS参数: %s ----\n', param_to_text(fs_param));
    dataset_sel = prepare_property_dataset(property_name, 'selected', preproc_mode, sg_order, sg_window, 'cars', fs_param, msc_ref_mode, snv_mode);
    for j = 1:numel(regressors)
        method_name = regressors{j};
        params = regressor_params.(method_name);
        for k = 1:numel(params)
            step_id = step_id + 1;
            method_param = params{k};
            fprintf('\n[进度 %d/%d] 分组2 | CARS=%s | 模型=%s | 参数=%s\n', ...
                step_id, step_total, param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param));
            result = train_model_from_dataset(dataset_sel.paths.mat, method_name, method_param);
            print_eta(step_id, step_total, global_tic);
            all_rows(end + 1, :) = build_report_row('SG+MSC+SNV + CARS筛选', 'cars', param_to_text(fs_param), get_model_display_name(method_name), param_to_text(method_param), result); %#ok<AGROW>
        end
    end
end

all_results_table = cell2table(all_rows, 'VariableNames', report_var_names());
all_results_table = sortrows(all_results_table, {'数据分组', '模型名称', 'R2_P', 'RMSEP'}, {'ascend', 'ascend', 'descend', 'ascend'});
summary_table = build_best_group_model_table(all_results_table);
report_top5 = build_top5_report(all_results_table);

save(fullfile(summary_dir, 'summary.mat'), 'summary_table', 'all_results_table', 'report_top5');
writetable(summary_table, fullfile(summary_dir, 'summary_best.csv'));
writetable(all_results_table, fullfile(summary_dir, 'all_results.csv'));
writetable(report_top5, fullfile(summary_dir, 'report_top5_by_algorithm.csv'));
try
    writetable(summary_table, fullfile(summary_dir, 'summary_best.xlsx'));
    writetable(all_results_table, fullfile(summary_dir, 'all_results.xlsx'));
    writetable(report_top5, fullfile(summary_dir, 'report_top5_by_algorithm.xlsx'));
catch
end

summary = summary_table;

fprintf('\n================ 两组数据回归器对比完成 ================\n');
fprintf('总耗时: %s\n', format_duration(toc(global_tic)));
disp(summary_table);
fprintf('完整结果已保存到: %s\n', summary_dir);
end

function row = build_report_row(group_name, fs_method, fs_param_text, model_display_name, train_param_text, result)
meta = result.dataset_metadata;
x_path = '';
y_path = '';
if isfield(meta, 'dataset_tag') && ~isempty(meta.dataset_tag)
    project_root = fileparts(mfilename('fullpath'));
    dataset_dir = fullfile(project_root, 'Result', 'Dataset', meta.dataset_tag);
    x_path = fullfile(dataset_dir, 'X.xlsx');
    y_path = fullfile(dataset_dir, 'y.xlsx');
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
    result.dataset_mat_path, ...
    x_path, ...
    y_path};
end

function names = report_var_names()
names = {'数据分组', '筛选方法', '筛选参数', '模型名称', '输入参数', '最优参数详情', ...
    '滤波方法', 'SG阶数', 'SG窗口', 'MSC模式', 'SNV模式', ...
    'R2_C', 'R2_P', 'RMSEC', 'RMSEP', 'RPD', ...
    '数据集MAT路径', 'X路径', 'Y路径'};
end

function summary_table = build_best_group_model_table(all_results_table)
rows = {};
group_col = all_results_table{:, '数据分组'};
model_col = all_results_table{:, '模型名称'};
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
summary_table = sortrows(summary_table, {'数据分组', 'R2_P', 'RMSEP'}, {'ascend', 'descend', 'ascend'});
end

function report_top5 = build_top5_report(all_results_table)
rows = {};
model_col = all_results_table{:, '模型名称'};
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
else
    txt = 'custom';
end
end

function print_eta(step_id, step_total, global_tic)
elapsed_sec = toc(global_tic);
avg_sec = elapsed_sec / max(step_id, 1);
remain_steps = step_total - step_id;
remain_sec = avg_sec * remain_steps;
fprintf('已耗时: %s | 预计剩余: %s\n', format_duration(elapsed_sec), format_duration(remain_sec));
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
    otherwise
        name = upper(method);
end
end

