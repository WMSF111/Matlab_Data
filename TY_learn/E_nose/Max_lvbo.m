%% ========================
%% 电子鼻：批量求每个传感器最大值 → 合成一个总表
%% 兼容所有CSV，不画图、纯计算、无维度错误
%% ========================
clear; clc; close all;

%% 1. 只改这里！你的文件夹
folder = '2026_3_24_2\偶数';  

%% 2. 获取所有CSV
fileList = dir(fullfile(folder, '*.csv'));
if isempty(fileList)
    error('没有找到CSV文件！');
end

%% 3. 初始化
allMaxData = [];
allFileNames = {};

%% 4. 遍历所有文件
for i = 1:length(fileList)
    fname = fullfile(folder, fileList(i).name);
    fprintf('处理：%s\n', fileList(i).name);
    
    % 读取
    T = readtable(fname);
    
    % 删除第一列
    T(:,1) = [];
    
    % 数据
    data = table2array(T);
    
    % 滤波（移动平均）
    for c = 1:size(data,2)
        data(:,c) = movmean(data(:,c), 5);
    end
    
    % 每个传感器最大值
    maxVals = max(data);
    
    % 保存
    allMaxData = [allMaxData; maxVals];
    allFileNames = [allFileNames; fileList(i).name];
end

%% 5. 读取第一个文件的传感器名（保证列名一致）
T_first = readtable(fullfile(folder, fileList(1).name));
T_first(:,1) = [];
sensorNames = T_first.Properties.VariableNames;

%% 6. 输出表格
T_out = array2table(allMaxData, 'VariableNames', sensorNames);
T_out.FileName = allFileNames;

% 把文件名放到第一列
T_out = movevars(T_out, 'FileName', 'Before', 1);

%% 7. 保存Excel
% writetable(T_out, '所有文件所有传感器最大值.xlsx');

fprintf('\n✅ 全部完成！\n');
fprintf('✅ 输出：所有文件所有传感器最大值.xlsx\n');

%% ========================
%% 🎯 只画【整体最大的2个传感器】
%% ========================
% 计算所有文件的平均最大值，找出最大的两个
meanMax = mean(allMaxData);
[~, idxSort] = sort(meanMax, 'descend');
top2_idx = idxSort(1:2);
top2_names = sensorNames(top2_idx);

fprintf('\n画图只显示最大的2个传感器：\n');
disp(top2_names);

% ======================
% 计算差值绝对值
% ======================
diff_abs = abs(allMaxData(:, top2_idx(1)) - allMaxData(:, top2_idx(2)));

% 绘图
figure('Color','w','Position',[100,100,800,400])
hold on;

% 画两个最大传感器
plot(allMaxData(:, top2_idx), 'o-', 'LineWidth',2);

% 画差值绝对值
plot(diff_abs, 's-', 'LineWidth',2, 'Color','k'); % 黑色虚线

title('所有文件最大值对比（仅显示最大2个传感器 + 差值绝对值）');
xlabel('文件编号');
ylabel('最大值 / 绝对差值');
legend([top2_names{1}, top2_names{2}, '差值绝对值'], 'Location','best');
grid on;
hold off;