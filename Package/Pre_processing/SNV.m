%% 标准正态变量
% 输出：snv_data 处理完数据
% 输入：data 处理前数据
function [snv_data]=SNV(data)
n = size(data,2);
X = mean(data,2);
dX = data - repmat(X,1,n);
snv_data = dX./repmat(sqrt(sum(dX.^2,2)/(n-1)),1,n);
end