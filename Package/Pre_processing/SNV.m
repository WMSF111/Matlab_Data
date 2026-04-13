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
        center_val = mean(data, 2);
        dX = data - repmat(center_val, 1, n);
        scale_val = sqrt(sum(dX.^2, 2) / max(n - 1, 1));

    case 'robust'
        center_val = median(data, 2);
        dX = data - repmat(center_val, 1, n);
        mad_val = median(abs(dX), 2);
        scale_val = 1.4826 * mad_val;

    otherwise
        error('不支持的 SNV 模式：%s。可选：standard / robust', snv_mode);
end

scale_val(scale_val < eps0) = 1;
snv_data = dX ./ repmat(scale_val, 1, n);
end
