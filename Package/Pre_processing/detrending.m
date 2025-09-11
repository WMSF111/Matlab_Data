%% 去趋势算法
% 输出：detrend_data 处理完数据
% 输入：data 处理前数据
% n = 0 删除均值 n = 1 去除线性趋势 n = 2 去除二次趋势。​
function [detrend_data]=detrending(data,n)
detrend_data = detrend(data,n);
end