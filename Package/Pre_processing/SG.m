%% SG平滑数据处理
% 输出：sg_data 处理完数据
% 输入
% data 处理前数据
% Order 阶数
% Framelen 窗口长度
function [sg_data] = SG(data,Order,Framelen)
sg_data = sgolayfilt(data,Order,Framelen);  
end

