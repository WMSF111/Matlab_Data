function model_result = train_model_from_dataset(dataset_mat_path, method_name, method_param)
% 功能：从已保存的数据集训练回归模型
% 支持回归器：
%   'pls' - 偏最小二乘回归
%   'pcr' - 主成分回归
%   'svr' - 支持向量回归
%   'rf'  - 随机森林回归
%
% 兼容旧入口：
%   'spa' / 'cars' / 'pca' / 'rfe'

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
result_dir = fullfile(project_root, 'Result', 'Model', 'DatasetTrain', dataset_tag, upper(method));
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
            tmp.PCRK = k;
            tmp.best_param_detail = sprintf('主成分数k=%d', k);
            if tmp.R2_P > max_R2_P
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
            tmp.SVRKernel = cfg.kernel;
            tmp.SVRBox = cfg.box;
            tmp.SVRScale = cfg.scale;
            tmp.best_param_detail = sprintf('核函数=%s, BoxConstraint=%g, KernelScale=%s', cfg.kernel, cfg.box, scale_to_text(cfg.scale));
            if tmp.R2_P > max_R2_P
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
            tmp.RFNumTrees = cfg.num_trees;
            tmp.RFMinLeaf = cfg.min_leaf;
            tmp.best_param_detail = sprintf('树数=%d, 最小叶子节点=%d', cfg.num_trees, cfg.min_leaf);
            if tmp.R2_P > max_R2_P
                max_R2_P = tmp.R2_P;
                model_result = tmp;
                fprintf('RF 当前最优更新: %s, R2P=%.4f, RMSEP=%.4f\n', tmp.best_param_detail, tmp.R2_P, tmp.RMSEP);
            end
        end

    case 'spa'
        k_range = [1, min(size(Xtrain, 2), 20)];
        if ~isempty(method_param)
            if numel(method_param) == 1
                k_range = [1, min(size(Xtrain, 2), round(method_param))];
            else
                k_range = [max(1, round(method_param(1))), min(size(Xtrain, 2), round(method_param(2)))];
            end
        end
        max_R2_P = -inf;
        model_result = struct();
        total_iter = size(Xtrain, 2) * (k_range(2) - k_range(1) + 1);
        iter_id = 0;
        fprintf('SPA-PLS 搜索开始: 起点特征=%d, k范围=%d~%d, 总组合=%d\n', size(Xtrain, 2), k_range(1), k_range(2), total_iter);
        for s = 1:size(Xtrain, 2)
            for k = k_range(1):k_range(2)
                iter_id = iter_id + 1;
                if mod(iter_id, 20) == 0 || iter_id == 1 || iter_id == total_iter
                    fprintf('SPA-PLS 进度: %d/%d\n', iter_id, total_iter);
                end
                wave_select = SPA(Xtrain, s, k);
                Xtrain_use = Xtrain(:, wave_select);
                Xtest_use = Xtest(:, wave_select);
                CV = plscv(Xtrain_use, Ytrain, 256, 10, 'center');
                A = CV.optLV;
                PLS = pls(Xtrain_use, Ytrain, A, 'center');
                Ypred = plsval(PLS, Xtest_use, Ytest);
                Ypred_train = plsval(PLS, Xtrain_use, Ytrain);
                tmp = build_result_struct(method, A, wave_select, select, Xtrain_use, Ytrain, Xtest_use, Ytest, Ypred_train, Ypred, CV);
                tmp.SPAS = s;
                tmp.SPAK = k;
                tmp.best_param_detail = sprintf('起始特征s=%d, 特征数k=%d, PLS潜变量A=%d', s, k, A);
                if tmp.R2_P > max_R2_P
                    max_R2_P = tmp.R2_P;
                    model_result = tmp;
                    fprintf('SPA-PLS 当前最优更新: s=%d, k=%d, A=%d, R2P=%.4f, RMSEP=%.4f\n', s, k, A, tmp.R2_P, tmp.RMSEP);
                end
            end
        end

    case 'cars'
        num_sampling_runs = 30;
        if ~isempty(method_param)
            num_sampling_runs = max(1, round(method_param));
        end
        fprintf('CARS-PLS 开始: CARS采样次数=%d\n', num_sampling_runs);
        CV0 = plscv(Xtrain, Ytrain, 256, 10, 'center');
        A0 = CV0.optLV;
        CARS = carspls(Xtrain, Ytrain, A0, 10, 'center', num_sampling_runs, 0, 1);
        selected_idx = CARS.vsel;
        Xtrain_use = Xtrain(:, selected_idx);
        Xtest_use = Xtest(:, selected_idx);
        CV = plscv(Xtrain_use, Ytrain, 256, 10, 'center');
        A = CV.optLV;
        PLS = pls(Xtrain_use, Ytrain, A, 'center');
        Ypred = plsval(PLS, Xtest_use, Ytest);
        Ypred_train = plsval(PLS, Xtrain_use, Ytrain);
        model_result = build_result_struct(method, A, selected_idx, select, Xtrain_use, Ytrain, Xtest_use, Ytest, Ypred_train, Ypred, CV);
        model_result.CARS = CARS;
        model_result.best_param_detail = sprintf('CARS采样次数=%d, 选中特征数=%d, PLS潜变量A=%d', num_sampling_runs, numel(selected_idx), A);

    case 'pca'
        [coeff, ~, ~, ~, ~, ~] = pca(Xtrain);
        max_R2_P = -inf;
        model_result = struct();
        total_iter = min(200, size(coeff, 2));
        fprintf('PCA-PLS 搜索开始: 主成分数量范围=1~%d\n', total_iter);
        for k = 1:total_iter
            if mod(k, 20) == 0 || k == 1 || k == total_iter
                fprintf('PCA-PLS 进度: %d/%d\n', k, total_iter);
            end
            Xtrain_use = Xtrain * coeff(:, 1:k);
            Xtest_use = Xtest * coeff(:, 1:k);
            CV = plscv(Xtrain_use, Ytrain, 256, 10, 'center');
            A = CV.optLV;
            PLS = pls(Xtrain_use, Ytrain, A, 'center');
            Ypred = plsval(PLS, Xtest_use, Ytest);
            Ypred_train = plsval(PLS, Xtrain_use, Ytrain);
            tmp = build_result_struct(method, A, 1:k, select, Xtrain_use, Ytrain, Xtest_use, Ytest, Ypred_train, Ypred, CV);
            tmp.PCAK = k;
            tmp.best_param_detail = sprintf('主成分数k=%d, PLS潜变量A=%d', k, A);
            if tmp.R2_P > max_R2_P
                max_R2_P = tmp.R2_P;
                model_result = tmp;
                fprintf('PCA-PLS 当前最优更新: k=%d, A=%d, R2P=%.4f, RMSEP=%.4f\n', k, A, tmp.R2_P, tmp.RMSEP);
            end
        end

    case 'rfe'
        nfeatures = min(40, size(Xtrain, 2));
        if ~isempty(method_param)
            nfeatures = min(size(Xtrain, 2), max(1, round(method_param)));
        end
        fprintf('RFE-PLS 开始: 保留特征数=%d\n', nfeatures);
        fun = @(XTrain, YTrain, XTest, YTest) ...
            sqrt(sum((YTest - plsval(pls(XTrain, YTrain, size(XTrain, 2), 'center'), XTest, YTest)).^2) / size(XTest, 1));
        opts = statset('Display', 'iter', 'UseParallel', false);
        [selectedFeatures, ~] = sequentialfs(fun, Xtrain, Ytrain, ...
            'cv', 10, 'direction', 'forward', 'options', opts, 'nfeatures', nfeatures);
        selected_idx = find(selectedFeatures);
        Xtrain_use = Xtrain(:, selected_idx);
        Xtest_use = Xtest(:, selected_idx);
        CV = plscv(Xtrain_use, Ytrain, 256, 10, 'center');
        A = CV.optLV;
        PLS = pls(Xtrain_use, Ytrain, A, 'center');
        Ypred = plsval(PLS, Xtest_use, Ytest);
        Ypred_train = plsval(PLS, Xtrain_use, Ytrain);
        model_result = build_result_struct(method, A, selected_idx, select, Xtrain_use, Ytrain, Xtest_use, Ytest, Ypred_train, Ypred, CV);
        model_result.best_param_detail = sprintf('RFE保留特征数=%d, 实际选中特征数=%d, PLS潜变量A=%d', nfeatures, numel(selected_idx), A);

    otherwise
        error('Unsupported method_name: %s.', method_name);
end

model_result.dataset_mat_path = dataset_mat_path;
model_result.dataset_tag = dataset_tag;
model_result.dataset_metadata = dataset.metadata;
model_result.method_name = method;
model_result.model_display_name = display_name;

result_mat = fullfile(result_dir, sprintf('Results_R2_P=%.4f.mat', model_result.R2_P));
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
saveas(fig, fullfile(result_dir, sprintf('Results_R2_P=%.4f.tif', model_result.R2_P)), 'tiff');
close(fig);

fprintf('训练完成: 模型=%s, R2P=%.4f, RMSEP=%.4f\n', display_name, model_result.R2_P, model_result.RMSEP);
fprintf('最优参数: %s\n', model_result.best_param_detail);
fprintf('模型结果目录：%s\n', result_dir);
end

function result = build_result_struct(method_name, A, selected_info, select_idx, Xtrain, Ytrain, Xtest, Ytest, Ypred_train, Ypred, CV)
SST_train = sum((Ytrain - mean(Ytrain)).^2);
SSE_train = sum((Ytrain - Ypred_train).^2);
SST_test = sum((Ytest - mean(Ytest)).^2);
SSE_test = sum((Ytest - Ypred).^2);

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
result.R2_C = 1 - SSE_train / SST_train;
result.R2_P = 1 - SSE_test / SST_test;
result.RMSEC = sqrt(SSE_train / size(Xtrain, 1));
result.RMSEP = sqrt(SSE_test / size(Xtest, 1));
if isempty(CV)
    result.RMSECV = NaN;
    result.Q2_Max = NaN;
else
    result.RMSECV = CV.RMSECV_min;
    result.Q2_Max = CV.Q2_max;
end
result.RPD = std(Ytest) / result.RMSEP;
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
    case 'spa'
        name = 'SPA-PLS';
    case 'cars'
        name = 'CARS-PLS';
    case 'pca'
        name = 'PCA-PLS';
    case 'rfe'
        name = 'RFE-PLS';
    otherwise
        name = upper(method);
end
end


