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
[select, not_select] = KS(X, ks_count);
Xtrain = X(select, :); Ytrain = y(select, :);
Xtest = X(not_select, :); Ytest = y(not_select, :);

switch method
    case 'pls'
        CV = plscv(Xtrain, Ytrain, 256, 10, 'center');
        A = CV.optLV;
        PLS = pls(Xtrain, Ytrain, A, 'center');
        Ypred = plsval(PLS, Xtest, Ytest);
        Ypred_train = plsval(PLS, Xtrain, Ytrain);
        model_result = build_result_struct(method, A, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, CV);
        model_result.best_param_detail = sprintf('PLS潜变量A=%d', A);

    case 'pcr'
        max_pc = min([200, size(Xtrain, 2), size(Xtrain, 1) - 1]);
        if ~isempty(method_param)
            max_pc = min(max_pc, max(1, round(method_param)));
        end
        [coeff, score, ~, ~, ~, mu] = pca(Xtrain);
        max_R2_P = -inf;
        model_result = struct();
        fprintf('PCR 搜索开始: 主成分数量范围=1~%d\n', max_pc);
        for k = 1:max_pc
            if mod(k, 20) == 0 || k == 1 || k == max_pc
                fprintf('PCR 进度: %d/%d\n', k, max_pc);
            end
            score_train = score(:, 1:k);
            score_test = bsxfun(@minus, Xtest, mu) * coeff(:, 1:k);
            beta = [ones(size(score_train, 1), 1), score_train] \ Ytrain;
            Ypred_train = [ones(size(score_train, 1), 1), score_train] * beta;
            Ypred = [ones(size(score_test, 1), 1), score_test] * beta;
            tmp = build_result_struct(method, k, 1:k, select, score_train, Ytrain, score_test, Ytest, Ypred_train, Ypred, []);
            tmp.best_param_detail = sprintf('主成分数k=%d', k);
            if ~isfield(model_result, 'R2_P') || is_better_result(tmp, model_result)
                max_R2_P = tmp.R2_P;
                model_result = tmp;
                fprintf('PCR 当前最优更新: k=%d, R2P=%.4f, RMSEP=%.4f\n', k, tmp.R2_P, tmp.RMSEP);
            end
        end

    case 'svr'
        grid = build_svr_grid(method_param);
        max_R2_P = -inf;
        model_result = struct();
        fprintf('SVR 搜索开始: 总组合=%d\n', numel(grid));
        for i = 1:numel(grid)
            cfg = grid(i);
            fprintf('SVR 进度: %d/%d | 核函数=%s | Box=%g | Scale=%s\n', i, numel(grid), cfg.kernel, cfg.box, scale_to_text(cfg.scale));
            try
                mdl = fitrsvm(Xtrain, Ytrain, ...
                    'KernelFunction', cfg.kernel, ...
                    'BoxConstraint', cfg.box, ...
                    'KernelScale', cfg.scale, ...
                    'Standardize', true);
            catch ME
                error('SVR 训练失败，请确认已安装 Statistics and Machine Learning Toolbox。原始错误: %s', ME.message);
            end
            Ypred_train = predict(mdl, Xtrain);
            Ypred = predict(mdl, Xtest);
            tmp = build_result_struct(method, NaN, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
            tmp.best_param_detail = sprintf('核函数=%s, BoxConstraint=%g, KernelScale=%s', cfg.kernel, cfg.box, scale_to_text(cfg.scale));
            if ~isfield(model_result, 'R2_P') || is_better_result(tmp, model_result)
                max_R2_P = tmp.R2_P;
                model_result = tmp;
                fprintf('SVR 当前最优更新: %s, R2P=%.4f, RMSEP=%.4f\n', tmp.best_param_detail, tmp.R2_P, tmp.RMSEP);
            end
        end

    case 'rf'
        grid = build_rf_grid(method_param);
        max_R2_P = -inf;
        model_result = struct();
        rng(1);
        fprintf('RF 搜索开始: 总组合=%d\n', numel(grid));
        for i = 1:numel(grid)
            cfg = grid(i);
            fprintf('RF 进度: %d/%d | 树数=%d | 叶子节点=%d\n', i, numel(grid), cfg.num_trees, cfg.min_leaf);
            try
                mdl = TreeBagger(cfg.num_trees, Xtrain, Ytrain, ...
                    'Method', 'regression', ...
                    'MinLeafSize', cfg.min_leaf, ...
                    'OOBPrediction', 'off');
            catch ME
                error('RF 训练失败，请确认已安装 Statistics and Machine Learning Toolbox。原始错误: %s', ME.message);
            end
            Ypred_train = predict(mdl, Xtrain);
            Ypred = predict(mdl, Xtest);
            if iscell(Ypred_train), Ypred_train = str2double(Ypred_train); end
            if iscell(Ypred), Ypred = str2double(Ypred); end
            tmp = build_result_struct(method, NaN, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train(:), Ypred(:), []);
            tmp.best_param_detail = sprintf('树数=%d, 最小叶子节点=%d', cfg.num_trees, cfg.min_leaf);
            if ~isfield(model_result, 'R2_P') || is_better_result(tmp, model_result)
                max_R2_P = tmp.R2_P;
                model_result = tmp;
                fprintf('RF 当前最优更新: %s, R2P=%.4f, RMSEP=%.4f\n', tmp.best_param_detail, tmp.R2_P, tmp.RMSEP);
            end
        end

    case 'gpr'
        grid = build_gpr_grid(method_param);
        max_R2_P = -inf;
        model_result = struct();
        fprintf('GPR 搜索开始: 总组合=%d\n', numel(grid));
        for i = 1:numel(grid)
            cfg = grid(i);
            fprintf('GPR 进度: %d/%d | 核函数=%s\n', i, numel(grid), cfg.kernel);
            try
                mdl = fitrgp(Xtrain, Ytrain, ...
                    'KernelFunction', cfg.kernel, ...
                    'Standardize', true);
            catch ME
                error('GPR 训练失败，请确认已安装 Statistics and Machine Learning Toolbox。原始错误: %s', ME.message);
            end
            Ypred_train = predict(mdl, Xtrain);
            Ypred = predict(mdl, Xtest);
            tmp = build_result_struct(method, NaN, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
            tmp.best_param_detail = sprintf('核函数=%s', cfg.kernel);
            if ~isfield(model_result, 'R2_P') || is_better_result(tmp, model_result)
                max_R2_P = tmp.R2_P;
                model_result = tmp;
                fprintf('GPR 当前最优更新: %s, R2P=%.4f, RMSEP=%.4f\n', tmp.best_param_detail, tmp.R2_P, tmp.RMSEP);
            end
        end

    case 'knn'
        grid = build_knn_grid(method_param);
        max_R2_P = -inf;
        model_result = struct();
        fprintf('KNN 搜索开始: 总组合=%d\n', numel(grid));
        for i = 1:numel(grid)
            cfg = grid(i);
            fprintf('KNN 进度: %d/%d | 邻居数=%d | 距离=%s | 加权=%s\n', i, numel(grid), cfg.k, cfg.distance, cfg.weighting);
            Ypred_train = local_knn_predict(Xtrain, Ytrain, Xtrain, cfg.k, cfg.distance, cfg.weighting, true);
            Ypred = local_knn_predict(Xtrain, Ytrain, Xtest, cfg.k, cfg.distance, cfg.weighting, false);
            tmp = build_result_struct(method, NaN, [], select, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, []);
            tmp.best_param_detail = sprintf('邻居数=%d, 距离=%s, 加权=%s', cfg.k, cfg.distance, cfg.weighting);
            if ~isfield(model_result, 'R2_P') || is_better_result(tmp, model_result)
                max_R2_P = tmp.R2_P;
                model_result = tmp;
                fprintf('KNN 当前最优更新: %s, R2P=%.4f, RMSEP=%.4f\n', tmp.best_param_detail, tmp.R2_P, tmp.RMSEP);
            end
        end

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
save(result_mat, 'model_result');

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

fprintf('训练完成: 模型=%s, R2P=%.4f, RMSEP=%.4f\n', display_name, model_result.R2_P, model_result.RMSEP);
fprintf('最优参数: %s\n', model_result.best_param_detail);
fprintf('模型结果目录：%s\n', result_dir);
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
else
    txt = 'custom';
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

