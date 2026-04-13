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
        ref = mean(data, 1);
    case 'median'
        ref = median(data, 1);
    case 'first'
        ref = data(1, :);
    otherwise
        error('不支持的 MSC 参考模式：%s。可选：mean / median / first', reference_mode);
end

MSC_data = zeros(size(data));
for i = 1:size(data, 1)
    p = polyfit(ref, data(i, :), 1);
    if abs(p(1)) < 1e-12
        MSC_data(i, :) = data(i, :) - p(2);
    else
        MSC_data(i, :) = (data(i, :) - p(2)) ./ p(1);
    end
end
end
