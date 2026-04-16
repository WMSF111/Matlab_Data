function model_result = train_model_from_dataset(dataset_mat_path, method_name, method_param)
% 功能：从已保存的数据集训练回归模型
% 支持回归器：
%   'pls' - 偏最小二乘回归
%   'pcr' - 主成分回归
%   'svr' - 支持向量回归
%   'rf'  - 随机森林回归
%   'gpr' - 高斯过程回归
%   'knn' - 自定义K近邻回归

if nargin < 1 || isempty(dataset_mat_path)
    error('请提供 dataset.mat 路径。');
end
if nargin < 2 || isempty(method_name)
    method_name = 'pls';
end
if nargin < 3
    method_param = [];
end

S = load(dataset_mat_path, 'dataset');
if ~isfield(S, 'dataset')
    error('数据集文件缺少变量 dataset: %s', dataset_mat_path);
end

dataset = S.dataset;
X = dataset.X;
y = dataset.y(:);

if size(X, 1) ~= numel(y)
    error('数据集 X 与 y 的样本数不一致。');
end

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

sample_count = size(X, 1);
ks_count = max(2, round(sample_count * 0.75));
method = lower(strtrim(method_name));
display_name = get_model_display_name(method);
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

fprintf('开始训练: 数据集=%s | 模型=%s | 参数=%s\n', strrep(dataset_tag, '\', ' / '), display_name, param_to_text(method_param));
[X_all_pre, ~, dataset.metadata] = local_preprocess_split(X, X, dataset.metadata);
dataset.metadata.preprocess_applied_on_all_samples = true;
dataset.metadata.preprocess_applied_after_split = false;
dataset.metadata.preprocess_tag = dataset_tag;
local_save_smooth_results(project_root, dataset_tag, X, X_all_pre, dataset.metadata);
X = X_all_pre;
[select, not_select] = KS(X, ks_count);
Xtrain = X(select, :); Ytrain = y(select, :);
Xtest = X(not_select, :); Ytest = y(not_select, :);
if isfield(dataset, 'metadata') && isfield(dataset.metadata, 'data_stage') && strcmpi(dataset.metadata.data_stage, 'selected')
    fprintf('全样本特征筛选开始: 方法=%s | 参数=%s\n', upper(char(dataset.metadata.fs_method)), param_to_text(dataset.metadata.fs_param));
    fprintf('注意：当前特征筛选使用训练集+测试集共同选特征，结果会更乐观。\n');
    Xall_fs = [Xtrain; Xtest];
    Yall_fs = [Ytrain; Ytest];
    fs_result = select_features_by_method(Xall_fs, Yall_fs, dataset.metadata.fs_method, dataset.metadata.fs_param);
    selected_idx = fs_result.selected_idx(:)';
    Xtrain = Xtrain(:, selected_idx);
    Xtest = Xtest(:, selected_idx);
    dataset.metadata.selected_idx = selected_idx;
    dataset.metadata.feature_score = fs_result.score;
    dataset.metadata.feature_info = fs_result.info;
    dataset.metadata.selection_scope = 'all_samples';
    if isfield(dataset.metadata, 'used_band_idx') && ~isempty(dataset.metadata.used_band_idx)
        base_band_idx = dataset.metadata.used_band_idx(:)';
        dataset.metadata.used_band_idx = base_band_idx(selected_idx);
        dataset.metadata.used_band_range = [min(dataset.metadata.used_band_idx), max(dataset.metadata.used_band_idx)];
    end
    fprintf('全样本特征筛选完成: 选中特征数=%d\n', numel(selected_idx));
end


switch method
    case 'pls'
        max_lv = min(256, size(Xtrain, 2));
        cv_fold = 10;
        if isstruct(method_param)
            if isfield(method_param, 'max_lv') && ~isempty(method_param.max_lv)
                max_lv = min(max(1, round(method_param.max_lv)), size(Xtrain, 2));
            end
            if isfield(method_param, 'cv_fold') && ~isempty(method_param.cv_fold)
                cv_fold = max(2, round(method_param.cv_fold));
            end
        end
        CV = plscv(Xtrain, Ytrain, max_lv, cv_fold, 'center');
        A = CV.optLV;
        PLS = pls(Xtrain, Ytrain, A, 'center');
        Ypred = plsval(PLS, Xtest, Ytest);
        Ypred_train = plsval(PLS, Xtrain, Ytrain);
        model_result = build_result_struct(method, A, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, CV);
        model_result.best_param_detail = sprintf('PLS潜变量A=%d, maxLV=%d, CV折数=%d', A, max_lv, cv_fold);

    case 'pcr'
        max_pc = min([200, size(Xtrain, 2), size(Xtrain, 1) - 1]);
        if isnumeric(method_param) && ~isempty(method_param)
            max_pc = min(max_pc, max(1, round(method_param)));
        elseif isstruct(method_param)
            if isfield(method_param, 'max_pc') && ~isempty(method_param.max_pc)
                max_pc = min(max_pc, max(1, round(method_param.max_pc)));
            end
        end
        [coeff, score, ~, ~, ~, mu] = pca(Xtrain);
        best_cv_rmse = inf;
        model_result = struct();
        best_k = 1;
        best_beta = [];
        fprintf('PCR 搜索开始: 主成分数量范围=1~%d（训练集内部CV选参）\n', max_pc);
        for k = 1:max_pc
            if mod(k, 20) == 0 || k == 1 || k == max_pc
                fprintf('PCR 进度: %d/%d\n', k, max_pc);
            end
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
        model_result = build_result_struct(method, best_k, 1:best_k, select, score_train, Ytrain, score_test, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = sprintf('主成分数k=%d, 训练集CV_RMSE=%.4f', best_k, best_cv_rmse);
        fprintf('PCR 最终选择: k=%d, 训练集CV_RMSE=%.4f, R2P=%.4f, RMSEP=%.4f\n', best_k, best_cv_rmse, model_result.R2_P, model_result.RMSEP);
    case 'svr'
        grid = build_svr_grid(method_param);
        best_cv_rmse = inf;
        model_result = struct();
        best_cfg = [];
        fprintf('SVR 搜索开始: 总组合=%d（训练集内部CV选参）\n', numel(grid));
        for i = 1:numel(grid)
            cfg = grid(i);
            fprintf('SVR 进度: %d/%d | 核函数=%s | Box=%g | Scale=%s\n', i, numel(grid), cfg.kernel, cfg.box, scale_to_text(cfg.scale));
            cv_rmse = local_svr_cv_rmse(Xtrain, Ytrain, cfg, 5);
            if cv_rmse < best_cv_rmse
                best_cv_rmse = cv_rmse;
                best_cfg = cfg;
                fprintf('SVR 当前最优更新: 核函数=%s, BoxConstraint=%g, KernelScale=%s, 训练集CV_RMSE=%.4f\n', cfg.kernel, cfg.box, scale_to_text(cfg.scale), cv_rmse);
            end
        end
        try
            mdl = fitrsvm(Xtrain, Ytrain, ...
                'KernelFunction', best_cfg.kernel, ...
                'BoxConstraint', best_cfg.box, ...
                'KernelScale', best_cfg.scale, ...
                'Standardize', true);
        catch ME
            error('SVR 训练失败，请确认已安装 Statistics and Machine Learning Toolbox。原始错误: %s', ME.message);
        end
        Ypred_train = predict(mdl, Xtrain);
        Ypred = predict(mdl, Xtest);
        model_result = build_result_struct(method, NaN, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = sprintf('核函数=%s, BoxConstraint=%g, KernelScale=%s, 训练集CV_RMSE=%.4f', best_cfg.kernel, best_cfg.box, scale_to_text(best_cfg.scale), best_cv_rmse);
        fprintf('SVR 最终选择: %s, R2P=%.4f, RMSEP=%.4f\n', model_result.best_param_detail, model_result.R2_P, model_result.RMSEP);

    case 'rf'
        grid = build_rf_grid(method_param);
        best_cv_rmse = inf;
        model_result = struct();
        best_cfg = [];
        rng(1);
        fprintf('RF 搜索开始: 总组合=%d（训练集内部CV选参）\n', numel(grid));
        for i = 1:numel(grid)
            cfg = grid(i);
            fprintf('RF 进度: %d/%d | 树数=%d | 叶子节点=%d\n', i, numel(grid), cfg.num_trees, cfg.min_leaf);
            cv_rmse = local_rf_cv_rmse(Xtrain, Ytrain, cfg, 5);
            if cv_rmse < best_cv_rmse
                best_cv_rmse = cv_rmse;
                best_cfg = cfg;
                fprintf('RF 当前最优更新: 树数=%d, 最小叶子节点=%d, 训练集CV_RMSE=%.4f\n', cfg.num_trees, cfg.min_leaf, cv_rmse);
            end
        end
        try
            mdl = TreeBagger(best_cfg.num_trees, Xtrain, Ytrain, ...
                'Method', 'regression', ...
                'MinLeafSize', best_cfg.min_leaf, ...
                'OOBPrediction', 'off');
        catch ME
            error('RF 训练失败，请确认已安装 Statistics and Machine Learning Toolbox。原始错误: %s', ME.message);
        end
        Ypred_train = predict(mdl, Xtrain);
        Ypred = predict(mdl, Xtest);
        if iscell(Ypred_train), Ypred_train = str2double(Ypred_train); end
        if iscell(Ypred), Ypred = str2double(Ypred); end
        model_result = build_result_struct(method, NaN, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train(:), Ypred(:), []);
        model_result.best_param_detail = sprintf('树数=%d, 最小叶子节点=%d, 训练集CV_RMSE=%.4f', best_cfg.num_trees, best_cfg.min_leaf, best_cv_rmse);
        fprintf('RF 最终选择: %s, R2P=%.4f, RMSEP=%.4f\n', model_result.best_param_detail, model_result.R2_P, model_result.RMSEP);


    case 'gpr'
        grid = build_gpr_grid(method_param);
        best_cv_rmse = inf;
        model_result = struct();
        best_cfg = [];
        fprintf('GPR 搜索开始: 总组合=%d（训练集内部CV选参）\n', numel(grid));
        for i = 1:numel(grid)
            cfg = grid(i);
            fprintf('GPR 进度: %d/%d | 核函数=%s\n', i, numel(grid), cfg.kernel);
            cv_rmse = local_gpr_cv_rmse(Xtrain, Ytrain, cfg, 5);
            if cv_rmse < best_cv_rmse
                best_cv_rmse = cv_rmse;
                best_cfg = cfg;
                fprintf('GPR 当前最优更新: 核函数=%s, 训练集CV_RMSE=%.4f\n', cfg.kernel, cv_rmse);
            end
        end
        try
            mdl = fitrgp(Xtrain, Ytrain, ...
                'KernelFunction', best_cfg.kernel, ...
                'Standardize', true);
        catch ME
            error('GPR 训练失败，请确认已安装 Statistics and Machine Learning Toolbox。原始错误: %s', ME.message);
        end
        Ypred_train = predict(mdl, Xtrain);
        Ypred = predict(mdl, Xtest);
        model_result = build_result_struct(method, NaN, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = sprintf('核函数=%s, 训练集CV_RMSE=%.4f', best_cfg.kernel, best_cv_rmse);
        fprintf('GPR 最终选择: %s, R2P=%.4f, RMSEP=%.4f\n', model_result.best_param_detail, model_result.R2_P, model_result.RMSEP);

    case 'knn'
        grid = build_knn_grid(method_param);
        best_cv_rmse = inf;
        model_result = struct();
        best_cfg = [];
        fprintf('KNN 搜索开始: 总组合=%d（训练集内部CV选参）\n', numel(grid));
        for i = 1:numel(grid)
            cfg = grid(i);
            fprintf('KNN 进度: %d/%d | 邻居数=%d | 距离=%s | 加权=%s\n', i, numel(grid), cfg.k, cfg.distance, cfg.weighting);
            cv_rmse = local_knn_cv_rmse(Xtrain, Ytrain, cfg, 5);
            if cv_rmse < best_cv_rmse
                best_cv_rmse = cv_rmse;
                best_cfg = cfg;
                fprintf('KNN 当前最优更新: 邻居数=%d, 距离=%s, 加权=%s, 训练集CV_RMSE=%.4f\n', cfg.k, cfg.distance, cfg.weighting, cv_rmse);
            end
        end
        Ypred_train = local_knn_predict(Xtrain, Ytrain, Xtrain, best_cfg.k, best_cfg.distance, best_cfg.weighting, true);
        Ypred = local_knn_predict(Xtrain, Ytrain, Xtest, best_cfg.k, best_cfg.distance, best_cfg.weighting, false);
        model_result = build_result_struct(method, NaN, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
        model_result.best_param_detail = sprintf('邻居数=%d, 距离=%s, 加权=%s, 训练集CV_RMSE=%.4f', best_cfg.k, best_cfg.distance, best_cfg.weighting, best_cv_rmse);
        fprintf('KNN 最终选择: %s, R2P=%.4f, RMSEP=%.4f\n', model_result.best_param_detail, model_result.R2_P, model_result.RMSEP);

    otherwise
        error('Unsupported method_name: %s.', method_name);
end

model_result.dataset_mat_path = dataset_mat_path;
model_result.dataset_tag = dataset_tag;
model_result.dataset_metadata = dataset.metadata;
model_result.method_name = method;
model_result.model_display_name = display_name;

result_stub = sprintf('%s_R2P_%.4f', safe_dataset_tag, model_result.R2_P);
result_mat = fullfile(result_dir, [result_stub '.mat']);
plot_path = fullfile(result_dir, [result_stub '.tif']);

fig = figure(301);
hold on;
plot(model_result.ytest2, model_result.ypred2, '.', 'MarkerSize', 15);
plot(model_result.Ytrain2, model_result.ypred2_train, '.', 'MarkerSize', 15);
plot(min(model_result.ytest2):max(model_result.ytest2), min(model_result.ytest2):max(model_result.ytest2), 'r-', 'LineWidth', 1.5);
hold off;
xlabel('Actual Values'); ylabel('Predicted Values');
title(sprintf('%s | R2C=%.3f, R2P=%.3f, RMSEP=%.3f', display_name, model_result.R2_C, model_result.R2_P, model_result.RMSEP));
legend('Test Data', 'Train Data', 'Ideal Line', 'Location', 'NorthWest');
saveas(fig, plot_path, 'tiff');
close(fig);

model_result.result_mat_path = result_mat;
model_result.regression_plot_path = plot_path;
selected_feature_plot_path = create_selected_feature_plot(project_root, dataset.metadata, result_dir, result_stub);
model_result.selected_feature_plot_path = selected_feature_plot_path;
bad_points_csv = fullfile(result_dir, [result_stub '_bad_points.csv']);
bad_points_table = build_bad_points_table(model_result, sample_count, project_root);
writetable(bad_points_table, bad_points_csv);
model_result.bad_points_csv_path = bad_points_csv;
model_result.bad_points_table = bad_points_table;
save(result_mat, 'model_result');

fprintf('训练完成: 模型=%s, R2P=%.4f, RMSEP=%.4f\n', display_name, model_result.R2_P, model_result.RMSEP);
fprintf('最优参数: %s\n', model_result.best_param_detail);
fprintf('模型结果目录：%s\n', result_dir);
fprintf('坏点清单：%s\n', bad_points_csv);
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

function result = build_result_struct(method_name, A, selected_info, select_idx, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, CV)
Ytrain = double(Ytrain(:));
Ytest = double(Ytest(:));
Ypred_train = double(Ypred_train(:));
Ypred = double(Ypred(:));

valid_train = isfinite(Ytrain) & isfinite(Ypred_train);
valid_test = isfinite(Ytest) & isfinite(Ypred);

SST_train = NaN; SSE_train = NaN; R2_C = NaN; RMSEC = NaN;
SST_test = NaN; SSE_test = NaN; R2_P = NaN; RMSEP = NaN; RPD = NaN;

if nnz(valid_train) >= 2
    ytr = Ytrain(valid_train);
    yptr = Ypred_train(valid_train);
    SST_train = sum((ytr - mean(ytr)).^2);
    SSE_train = sum((ytr - yptr).^2);
    RMSEC = sqrt(SSE_train / numel(ytr));
    if SST_train > eps
        R2_C = 1 - SSE_train / SST_train;
    end
end

if nnz(valid_test) >= 2
    yte = Ytest(valid_test);
    ypte = Ypred(valid_test);
    SST_test = sum((yte - mean(yte)).^2);
    SSE_test = sum((yte - ypte).^2);
    RMSEP = sqrt(SSE_test / numel(yte));
    if SST_test > eps
        R2_P = 1 - SSE_test / SST_test;
    end
    if isfinite(RMSEP) && RMSEP > 0
        RPD = std(yte) / RMSEP;
    end
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
if isempty(CV)
    result.RMSECV = NaN;
    result.Q2_Max = NaN;
else
    result.RMSECV = CV.RMSECV_min;
    result.Q2_Max = CV.Q2_max;
end
result.RPD = RPD;
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
elseif ischar(method_param) && strcmpi(method_param, 'fast')
    ks = [3, 5, 7];
    distances = {'euclidean'};
    weightings = {'uniform', 'inverse'};
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

function tf = is_better_result(candidate, current_best)
if ~isfield(current_best, 'R2_P') || isempty(current_best.R2_P)
    tf = true;
    return;
end
if isnan(current_best.R2_P) && ~isnan(candidate.R2_P)
    tf = true;
    return;
end
if isnan(candidate.R2_P)
    tf = false;
    return;
end
if candidate.R2_P > current_best.R2_P
    tf = true;
elseif candidate.R2_P < current_best.R2_P
    tf = false;
else
    tf = candidate.RMSEP < current_best.RMSEP;
end
end
function txt = scale_to_text(scale)
if ischar(scale)
    txt = scale;
else
    txt = num2str(scale);
end
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
elseif isstruct(param)
    txt = 'struct';
else
    txt = 'custom';
end
end


function T = build_bad_points_table(model_result, sample_count, project_root)
train_idx = model_result.Rank2(:);
all_idx = (1:sample_count)';
test_mask = true(sample_count, 1);
test_mask(train_idx) = false;
test_idx = all_idx(test_mask);

T_train = build_split_error_table('train', train_idx, model_result.Ytrain2, model_result.ypred2_train);
T_test = build_split_error_table('test', test_idx, model_result.ytest2, model_result.ypred2);
T = [T_train; T_test];

nir_dir = fullfile(project_root, 'data', 'NIR');
if exist(nir_dir, 'dir')
    try
        nir_files = dir(fullfile(nir_dir, '*.csv'));
        [~, idx_sort] = sort({nir_files.name});
        nir_files = nir_files(idx_sort);
        csv_names = string({nir_files.name})';
        T.csv_name = strings(height(T), 1);
        valid_idx = T.sample_index >= 1 & T.sample_index <= numel(csv_names);
        T.csv_name(valid_idx) = csv_names(T.sample_index(valid_idx));
        T = movevars(T, 'csv_name', 'After', 'sample_index');
    catch
    end
end

T = sortrows(T, {'abs_error', 'split'}, {'descend', 'ascend'});
end

function T = build_split_error_table(split_name, sample_idx, y_true, y_pred)
y_true = double(y_true(:));
y_pred = double(y_pred(:));
sample_idx = sample_idx(:);
n = min([numel(sample_idx), numel(y_true), numel(y_pred)]);
sample_idx = sample_idx(1:n);
y_true = y_true(1:n);
y_pred = y_pred(1:n);

abs_error = abs(y_pred - y_true);
den = max(abs(y_true), 1e-12);
rel_error = abs_error ./ den;
split_col = repmat({split_name}, n, 1);

T = table(split_col, sample_idx, y_true, y_pred, abs_error, rel_error, ...
    'VariableNames', {'split', 'sample_index', 'y_true', 'y_pred', 'abs_error', 'rel_error'});
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

function plot_path = create_selected_feature_plot(project_root, metadata, result_dir, result_stub)
plot_path = '';
if ~isfield(metadata, 'preprocess_tag') || isempty(metadata.preprocess_tag)
    return;
end
smooth_dir = fullfile(project_root, 'Result', 'Smooth', metadata.preprocess_tag);
smooth_mat = fullfile(smooth_dir, 'Smooth_Results.mat');
if ~exist(smooth_mat, 'file')
    return;
end
S = load(smooth_mat);
fs_method = 'UNKNOWN';
if isfield(metadata, 'fs_method') && ~isempty(metadata.fs_method)
    fs_method = lower(strtrim(char(metadata.fs_method)));
end
preproc_mode = 'unknown';
if isfield(metadata, 'preproc_mode') && ~isempty(metadata.preproc_mode)
    preproc_mode = char(metadata.preproc_mode);
end
if strcmpi(fs_method, 'pca') && isfield(metadata, 'feature_info') && isstruct(metadata.feature_info) && isfield(metadata.feature_info, 'coeff') && ~isempty(metadata.feature_info.coeff)
    coeff = metadata.feature_info.coeff;
    explained = [];
    if isfield(metadata.feature_info, 'explained'), explained = metadata.feature_info.explained; end
    n_pc = min([3, size(coeff, 2)]);
    if n_pc < 1
        return;
    end
    if isfield(S, 'x_axis') && numel(S.x_axis) == size(coeff, 1)
        x_axis = S.x_axis(:)';
    else
        x_axis = 1:size(coeff, 1);
    end
    fig = figure(302);
    tiledlayout(n_pc + 1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    for i = 1:n_pc
        nexttile; plot(x_axis, coeff(:, i), 'LineWidth', 1.1);
        if ~isempty(explained) && numel(explained) >= i
            title(sprintf('PC%d Loading | Explained = %.2f%%', i, explained(i)));
        else
            title(sprintf('PC%d Loading', i));
        end
    end
    nexttile;
    if isfield(metadata.feature_info, 'pca_score') && ~isempty(metadata.feature_info.pca_score)
        plot(x_axis, metadata.feature_info.pca_score(:), 'r-', 'LineWidth', 1.1);
        title('Weighted PCA Contribution');
    else
        plot(x_axis, sum(abs(coeff(:, 1:n_pc)), 2), 'r-', 'LineWidth', 1.1);
        title('Summed Absolute Loadings');
    end
    xlabel('Wavelength / Feature');
    plot_path = fullfile(result_dir, [result_stub '_pca_loadings.tif']);
    smooth_overlay_path = fullfile(smooth_dir, 'PCA_Loadings_On_Smooth.tif');
    sgtitle(sprintf('PCA | 主成分数=%d | 预处理方式=%s', n_pc, preproc_mode));
    saveas(fig, plot_path, 'tiff');
    saveas(fig, smooth_overlay_path, 'tiff');
    close(fig);
    return;
end
if ~isfield(metadata, 'selected_idx') || isempty(metadata.selected_idx)
    return;
end
if ~isfield(S, 'Post_smooth_data') || isempty(S.Post_smooth_data)
    return;
end
Xplot = S.Post_smooth_data;
mean_curve = mean(Xplot, 1, 'omitnan');
if isfield(S, 'x_axis') && numel(S.x_axis) == numel(mean_curve)
    x_axis = S.x_axis(:)';
else
    x_axis = 1:numel(mean_curve);
end
selected_idx = metadata.selected_idx(:)';
selected_idx = selected_idx(selected_idx >= 1 & selected_idx <= numel(mean_curve));
if isempty(selected_idx)
    return;
end
plot_path = fullfile(result_dir, [result_stub '_selected_points.tif']);
smooth_overlay_path = fullfile(smooth_dir, 'Selected_Points_On_Smooth.tif');
fig = figure(302);
plot(x_axis, mean_curve, 'b-', 'LineWidth', 1.0); hold on;
scatter(x_axis(selected_idx), mean_curve(selected_idx), 28, 'r', 'filled');
hold off;
xlabel('Wavelength / Feature');
ylabel('Intensity');
selected_count = numel(selected_idx);
title(sprintf('筛选方法: %s | 选中特征数: %d | 预处理方式: %s', upper(fs_method), selected_count, preproc_mode));
legend('Filtered Mean Spectrum', 'Selected Features', 'Location', 'best');
saveas(fig, plot_path, 'tiff');
saveas(fig, smooth_overlay_path, 'tiff');
close(fig);
end
function rmse = local_pcr_cv_rmse(score_train, y_train, nfold)
y_train = double(y_train(:));
n = size(score_train, 1);
if nargin < 3 || isempty(nfold)
    nfold = 5;
end
nfold = max(2, min(nfold, n));
fold_id = local_make_fold_id(n, nfold);
ypred = nan(n, 1);
for f = 1:nfold
    te = (fold_id == f);
    tr = ~te;
    Xtr = score_train(tr, :);
    Ytr = y_train(tr);
    Xte = score_train(te, :);
    beta = [ones(size(Xtr, 1), 1), Xtr] \ Ytr;
    ypred(te) = [ones(size(Xte, 1), 1), Xte] * beta;
end
valid = isfinite(y_train) & isfinite(ypred);
if nnz(valid) < 2
    rmse = inf;
else
    rmse = sqrt(mean((y_train(valid) - ypred(valid)).^2));
end
end
function rmse = local_svr_cv_rmse(Xtrain, Ytrain, cfg, nfold)
n = size(Xtrain, 1);
fold_id = local_make_fold_id(n, nfold);
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
fold_id = local_make_fold_id(n, nfold);
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
fold_id = local_make_fold_id(n, nfold);
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
fold_id = local_make_fold_id(n, nfold);
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
function fold_id = local_make_fold_id(n, nfold)
nfold = max(2, min(nfold, n));
rng(1);
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
    error('波段裁剪后没有剩余特征。');
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

stage = 'raw';
if isfield(meta, 'data_stage') && ~isempty(meta.data_stage), stage = lower(strtrim(meta.data_stage)); end
if strcmpi(stage, 'raw')
    Xtrain_out = Xtrain_work;
    Xtest_out = Xtest_work;
else
    [Xtrain_out, Xtest_out] = local_apply_preproc_pair(Xtrain_work, Xtest_work, preproc_mode, sg_order, sg_window, msc_ref_mode, snv_mode);
end
meta.used_band_idx = used_band_idx;
meta.used_band_range = [used_band_idx(1), used_band_idx(end)];
meta.preprocess_applied_after_split = true;
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
        error('不支持的 MSC 参考模式：%s。', reference_mode);
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
    slope_eps = max(1e-8, 1e-6 * std(row(valid)));
    if ~isfinite(p(1)) || abs(p(1)) < slope_eps
        corrected = row - p(2);
    else
        corrected = (row - p(2)) ./ p(1);
    end
    corrected(~isfinite(corrected)) = row(~isfinite(corrected));
    Xout(i, :) = corrected;
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
        n = min(5, size(X_in, 2));
        base = mean(X_in(:, 1:n), 2, 'omitnan');
    otherwise
        error('Unsupported baseline_zero_mode: %s.', mode);
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
    case 'local'
        X_out = local_despike_isolated_split(X_in, 3);
    case 'local_strong'
        X_out = local_despike_isolated_split(X_in, 2);
    case 'jump_guard'
        X_out = local_jump_guard_split(X_in);
    otherwise
        error('Unsupported despike_mode: %s.', mode);
end
end

function X_out = local_despike_isolated_split(X_in, thresh_scale)
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
        neigh_mean = (left_v + right_v) / 2;
        neigh_diff = abs(right_v - left_v);
        spike_mag = abs(mid_v - neigh_mean);
        local_scale = max([neigh_diff, abs(mid_v - left_v), abs(mid_v - right_v), eps]);
        if spike_mag > thresh_scale * local_scale && sign(mid_v - left_v) ~= sign(right_v - mid_v)
            row(c) = neigh_mean;
        end
    end
    X_out(r, :) = row;
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
        d1 = abs(mid_v - left_v);
        d2 = abs(mid_v - right_v);
        neigh_gap = abs(left_v - right_v);
        jump_ref = max([neigh_gap, eps]);
        if d1 > 4 * jump_ref && d2 > 4 * jump_ref
            row(c) = (left_v + right_v) / 2;
        end
    end
    X_out(r, :) = row;
end
end

function local_save_smooth_results(project_root, smooth_tag, X_raw, X_post, meta)
if nargin < 5 || isempty(meta)
    meta = struct();
end
if isempty(X_raw) || isempty(X_post)
    return;
end
smooth_mat = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.mat');
smooth_tif = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.tif');
smooth_fig = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.fig');
smooth_dir = fileparts(smooth_mat);
if ~exist(smooth_dir, 'dir')
    mkdir(smooth_dir);
end
preproc_mode = 'none';
if isfield(meta, 'preproc_mode') && ~isempty(meta.preproc_mode), preproc_mode = char(meta.preproc_mode); end
sg_order = 3;
if isfield(meta, 'sg_order') && ~isempty(meta.sg_order), sg_order = meta.sg_order; end
sg_window = 15;
if isfield(meta, 'sg_window') && ~isempty(meta.sg_window), sg_window = meta.sg_window; end
msc_ref_mode = 'mean';
if isfield(meta, 'msc_ref_mode') && ~isempty(meta.msc_ref_mode), msc_ref_mode = char(meta.msc_ref_mode); end
snv_mode = 'standard';
if isfield(meta, 'snv_mode') && ~isempty(meta.snv_mode), snv_mode = char(meta.snv_mode); end
cut_left = 0; if isfield(meta, 'cut_left') && ~isempty(meta.cut_left), cut_left = meta.cut_left; end
cut_right = 0; if isfield(meta, 'cut_right') && ~isempty(meta.cut_right), cut_right = meta.cut_right; end
baseline_zero_mode = 'none';
if isfield(meta, 'baseline_zero_mode') && ~isempty(meta.baseline_zero_mode), baseline_zero_mode = char(meta.baseline_zero_mode); end
baseline_zero_scope = 'cropped_spectrum';
if isfield(meta, 'baseline_zero_scope') && ~isempty(meta.baseline_zero_scope), baseline_zero_scope = char(meta.baseline_zero_scope); end
despike_mode = 'none';
if isfield(meta, 'despike_mode') && ~isempty(meta.despike_mode), despike_mode = char(meta.despike_mode); end
x_axis_full = local_load_plot_x_axis(project_root, size(X_raw, 2));
source_feature_count = size(X_raw, 2);
used_band_idx = (1 + cut_left):(source_feature_count - cut_right);
if isempty(used_band_idx)
    used_band_idx = 1:source_feature_count;
end
work_full = local_apply_despike_split(X_raw, despike_mode);
if strcmpi(strtrim(baseline_zero_scope), 'full_spectrum')
    work_full = local_apply_baseline_zero_split(work_full, baseline_zero_mode);
    apply_cropped_baseline = false;
else
    apply_cropped_baseline = true;
end
work_crop = work_full(:, used_band_idx);
x_axis_post = x_axis_full(used_band_idx);
raw_zero = work_crop;
if apply_cropped_baseline
    raw_zero = local_apply_baseline_zero_split(raw_zero, baseline_zero_mode);
end
SG_only = SG(work_crop, sg_order, sg_window);
if apply_cropped_baseline
    SG_only = local_apply_baseline_zero_split(SG_only, baseline_zero_mode);
end
MSC_only = local_msc_with_ref(work_crop, local_choose_msc_ref(work_crop, msc_ref_mode));
if apply_cropped_baseline
    MSC_only = local_apply_baseline_zero_split(MSC_only, baseline_zero_mode);
end
SNV_only = SNV(work_crop, snv_mode);
if apply_cropped_baseline
    SNV_only = local_apply_baseline_zero_split(SNV_only, baseline_zero_mode);
end
Post_smooth_data = X_post;
fig_smooth = figure(401);
subplot(2,3,1); plot(x_axis_post, raw_zero'); title('source data');
subplot(2,3,2); plot(x_axis_post, SG_only'); title('SG only');
subplot(2,3,3); plot(x_axis_post, MSC_only'); title(['MSC only (', msc_ref_mode, ')']);
subplot(2,3,4); plot(x_axis_post, SNV_only'); title(['SNV only (', snv_mode, ')']);
subplot(2,3,5); plot(x_axis_post, X_post'); title(['final used: ', preproc_mode]);
subplot(2,3,6); axis off;
x_axis = x_axis_post;
save(smooth_mat, 'Post_smooth_data', 'preproc_mode', 'sg_order', 'sg_window', 'msc_ref_mode', 'snv_mode', 'cut_left', 'cut_right', 'used_band_idx', 'x_axis', 'x_axis_post', 'baseline_zero_mode', 'baseline_zero_scope', 'despike_mode');
saveas(fig_smooth, smooth_tif, 'tiff');
savefig(fig_smooth, smooth_fig);
close(fig_smooth);
end

function ref = local_choose_msc_ref(Xin, reference_mode)
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
