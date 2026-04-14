function [MSC_data] = MSC(data, reference_mode)
%% 多元散射校正
% 输入：
%   data           - 原始数据（行=样本，列=特征）
%   reference_mode - 参考光谱模式：'mean' / 'median' / 'first'
% 输出：
%   MSC_data       - MSC 处理后的数据

if nargin < 2 || isempty(reference_mode)
    reference_mode = 'mean';
end

switch lower(strtrim(reference_mode))
    case 'mean'
        ref = mean(data, 1, 'omitnan');
    case 'median'
        ref = median(data, 1, 'omitnan');
    case 'first'
        ref = data(1, :);
    otherwise
        error('不支持的 MSC 参考模式：%s。可选：mean / median / first', reference_mode);
end

MSC_data = zeros(size(data));
for i = 1:size(data, 1)
    row = data(i, :);
    valid = isfinite(ref) & isfinite(row);
    if nnz(valid) < 2
        MSC_data(i, :) = row;
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
    MSC_data(i, :) = local_despike(corrected);
end
end

function row_out = local_despike(row_in)
row_out = row_in;
if numel(row_in) < 5
    return;
end

d = diff(row_in);
valid_d = d(isfinite(d));
if isempty(valid_d)
    return;
end

thr = median(abs(valid_d - median(valid_d))) * 6;
if ~isfinite(thr) || thr <= 0
    thr = std(valid_d) * 4;
end
if ~isfinite(thr) || thr <= 0
    return;
end

for j = 2:(numel(row_in) - 1)
    if ~isfinite(row_out(j - 1)) || ~isfinite(row_out(j)) || ~isfinite(row_out(j + 1))
        continue;
    end
    left_jump = abs(row_out(j) - row_out(j - 1));
    right_jump = abs(row_out(j) - row_out(j + 1));
    neighbor_gap = abs(row_out(j - 1) - row_out(j + 1));
    if left_jump > thr && right_jump > thr && neighbor_gap < thr
        row_out(j) = (row_out(j - 1) + row_out(j + 1)) / 2;
    end
end
end