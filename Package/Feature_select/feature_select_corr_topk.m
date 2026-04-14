% 功能：兼容保留版特征筛选（主实现已放到 Package/Feature_select）
% 说明：保留同名函数用于兼容旧路径调用，算法与 Package 版本一致。
function [selected_idx, score_abs] = feature_select_corr_topk(X, y, top_k)

if nargin < 3 || isempty(top_k)
    top_k = min(100, size(X, 2));
end

y = y(:);
if size(X, 1) ~= numel(y)
    error('X 和 y 的样本数量不一致。');
end

n_features = size(X, 2);
score_abs = zeros(1, n_features);

for j = 1:n_features
    xj = X(:, j);
    r = corr(xj, y, 'Type', 'Pearson', 'Rows', 'complete');
    if isnan(r)
        r = 0;
    end
    score_abs(j) = abs(r);
end

[~, order] = sort(score_abs, 'descend');
top_k = max(1, min(top_k, n_features));
selected_idx = order(1:top_k);
end
