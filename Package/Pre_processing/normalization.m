%% 归一化处理
% 输出：norm_data 处理完数据
% 输入：data 处理前数据
% mode 模式选择 0:矢量归一化 1：最大最小归一化 2：Z-score归一化
function [norm_data] = normalization(data,mode)
[m, n] = size(data);
norm_data = zeros(m, n);
if mode == 0
    mean_data = mean_centering(data); %每一行求均值
    s = sqrt(sum(data.*data));     %矢量归一化分子
    norm_data = mean_data ./ s;
elseif mode == 1
     ma = max( data,[],2);
     mi = min( data,[],2);
     norm_data = ( data-mi ) ./ ( ma-mi );
elseif mode == 2
     mea = mean( data,2);
     st = std( data,0,2);
     norm_data= ( data-mea ) ./ st;
end
end