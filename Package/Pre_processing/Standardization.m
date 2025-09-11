%% 标准化处理
% 输出：std_data 处理完数据
% 输入：data 处理前数据
function [std_data] = Standardization(data)
mean_data = mean_centering(data); %每一行求均值
s = std(data,0,2);             %行求标准差
std_data = mean_data ./ s;
end