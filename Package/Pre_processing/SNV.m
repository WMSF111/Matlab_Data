function [snv_data] = SNV(data, snv_mode)
%% 标准正态变量变换
% 输入：
%   data     - 原始数据（行=样本，列=特征）
%   snv_mode - 模式：'standard' / 'robust'
% 输出：
%   snv_data - SNV 处理后的数据

if nargin < 2 || isempty(snv_mode)
    snv_mode = 'standard';
end

n = size(data, 2);
eps0 = 1e-12;

switch lower(strtrim(snv_mode))
    case 'standard'
        center_val = mean(data, 2, 'omitnan');
        dX = data - repmat(center_val, 1, n);
        scale_val = sqrt(sum(dX.^2, 2, 'omitnan') / max(n - 1, 1));

    case 'robust'
        center_val = median(data, 2, 'omitnan');
        dX = data - repmat(center_val, 1, n);
        mad_val = median(abs(dX), 2, 'omitnan');
        scale_val = 1.4826 * mad_val;

    otherwise
        error('不支持的 SNV 模式：%s。可选：standard / robust', snv_mode);
end

finite_scale = scale_val(isfinite(scale_val));
if isempty(finite_scale)
    scale_floor = 1;
else
    scale_floor = max(eps0, median(finite_scale) * 1e-3);
end
if ~isfinite(scale_floor) || scale_floor <= 0
    scale_floor = 1;
end

scale_val(scale_val < scale_floor | ~isfinite(scale_val)) = scale_floor;
snv_data = dX ./ repmat(scale_val, 1, n);
snv_data(~isfinite(snv_data)) = 0;

for i = 1:size(snv_data, 1)
    snv_data(i, :) = local_despike(snv_data(i, :));
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