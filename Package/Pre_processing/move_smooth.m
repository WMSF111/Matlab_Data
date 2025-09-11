%% 移动平均处理
% 输出：move_data 处理完数据
% 输入：data 处理前数据 window 窗口大小（一般为奇数）
function [move_data]=move_smooth(data,window)
move_data = movmean(data, window, 2);
end


