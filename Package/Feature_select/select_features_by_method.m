function fs_result = select_features_by_method(X, y, fs_method, fs_param)
% 功能：统一特征筛选入口
% 输入：
%   X         - 样本矩阵（行=样本，列=特征）
%   y         - 目标向量
%   fs_method - 筛选算法：'corr_topk' / 'pca' / 'cars' / 'spa'
%   fs_param  - 算法参数
%               corr_topk/pca/spa: 选择特征数
%               cars: CARS 采样次数（默认 30）
% 输出：
%   fs_result.selected_idx - 选中特征索引
%   fs_result.score        - 特征重要性分数（无连续分数时为 0/1）
%   fs_result.info         - 算法附加信息
%
% 方法说明（按与训练耦合强弱由弱到强大致排序）：
%   1) pca
%      - 无监督训练前降维/筛选
%      - 只利用 X 的整体方差结构，不直接使用 y
%      - 更适合作为训练前的数据压缩或预筛选
%   2) corr_topk
%      - 监督式训练前筛选
%      - 利用每个特征与 y 的相关性排序，但不真正训练模型
%      - 适合作为训练前快速预筛选
%   3) spa
%      - 偏训练前筛选，但与建模有一定关系
%      - 重点是减少共线性、保留信息不重复的特征
%      - 当前实现里起始特征参考了与 y 的相关性，因此与任务有一定耦合
%   4) cars
%      - 建模驱动筛选，和训练最密切相关
%      - 基于 PLS/CARS 反复抽样得到更有利于建模的特征
%      - 更适合在明确建模目标后使用

if nargin < 3 || isempty(fs_method)
    fs_method = 'corr_topk';
end
if nargin < 4
    fs_param = [];
end

y = y(:);
if size(X, 1) ~= numel(y)
    error('X 与 y 的样本数不一致。');
end

n_features = size(X, 2);
method = lower(strtrim(fs_method));

fs_result = struct();
fs_result.method = method;
fs_result.selected_idx = [];
fs_result.score = zeros(1, n_features);
fs_result.info = struct();

switch method
    case 'corr_topk'
        top_k = normalize_param(fs_param, n_features, min(100, n_features));
        [selected_idx, score_abs] = feature_select_corr_topk(X, y, top_k);
        fs_result.selected_idx = selected_idx(:)';
        fs_result.score = score_abs(:)';
        fs_result.info.top_k = top_k;

    case 'pca'
        top_k = normalize_param(fs_param, n_features, min(30, n_features));
        [coeff, ~, latent, ~, explained] = pca(X);
        cum_explained = cumsum(explained);
        n_pc = find(cum_explained >= 99, 1, 'first');
        if isempty(n_pc)
            n_pc = min(size(coeff, 2), 1);
        end

        pc_weights = explained(1:n_pc);
        if sum(pc_weights) <= 0
            pc_weights = ones(n_pc, 1) / n_pc;
        else
            pc_weights = pc_weights / sum(pc_weights);
        end

        score = sum(abs(coeff(:, 1:n_pc)) .* reshape(pc_weights, 1, []), 2);
        [~, order] = sort(score, 'descend');

        fs_result.selected_idx = order(1:top_k)';
        fs_result.score = score(:)';
        fs_result.info.top_k = top_k;
        fs_result.info.n_pc = n_pc;
        fs_result.info.latent = latent;
        fs_result.info.explained = explained;

    case 'cars'
        num_sampling_runs = normalize_param(fs_param, inf, 30);
        CV = plscv(X, y, 256, 10, 'center');
        A = CV.optLV;
        CARS = carspls(X, y, A, 10, 'center', num_sampling_runs, 0, 1);
        selected_idx = CARS.vsel(:)';

        score = zeros(1, n_features);
        score(selected_idx) = 1;

        fs_result.selected_idx = selected_idx;
        fs_result.score = score;
        fs_result.info.A = A;
        fs_result.info.num_sampling_runs = num_sampling_runs;
        fs_result.info.CARS = CARS;

    case 'spa'
        top_k = normalize_param(fs_param, n_features, min(30, n_features));
        [~, corr_score] = feature_select_corr_topk(X, y, n_features);
        [~, initial_idx] = max(corr_score);
        selected_idx = SPA(X, initial_idx, top_k);

        score = zeros(1, n_features);
        score(selected_idx) = 1;

        fs_result.selected_idx = selected_idx(:)';
        fs_result.score = score;
        fs_result.info.top_k = top_k;
        fs_result.info.initial_idx = initial_idx;

    otherwise
        error('暂不支持的特征筛选方法：%s。可选：corr_topk / pca / cars / spa', fs_method);
end

fs_result.selected_idx = unique(fs_result.selected_idx, 'stable');
end

function out = normalize_param(value, upper_bound, default_value)
if nargin < 3 || isempty(default_value)
    default_value = 1;
end

if isempty(value)
    out = default_value;
else
    out = round(value);
end

out = max(1, out);
if isfinite(upper_bound)
    out = min(out, upper_bound);
end
end
