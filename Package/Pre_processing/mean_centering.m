%% 均值中心化处理
% 输出：mean_centering_data 处理完数据
% 输入：data 处理前数据
function [mean_centering_data] = mean_centering(data)
mean_centering = mean(data,2); %每一行求均值
mean_centering_data = data - mean_centering;
end