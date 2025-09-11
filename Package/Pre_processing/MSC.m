%% 多元散射校正算法
% 输出：MSC_data 处理完数据
% 输入：data 处理前数据
function [MSC_data] = MSC(data)
for i=1:size(data,1)
    p=polyfit(mean(data),data(i,:),1);%曲线拟合
    MSC_data(i,:)=(data(i,:)-p(2)*ones(1,size(data,2)))./(p(1)*ones(1,size(data,2)));
end 
end