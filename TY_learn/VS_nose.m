clear;
clc;
% ==========================================================
% 强制开启中文支持（修复中文方框乱码）
% ==========================================================
set(0,'DefaultAxesFontName','SimHei');  % 黑体
set(0,'DefaultTextFontName','SimHei');
set(0,'DefaultLegendFontName','SimHei');
% ==========================================================
% 1. 配置部分（已修改：e_nose 文件夹 + 4个 xlsx 文件）
% ==========================================================
dataFolder = 'E_nose\test';  % 你的文件夹名称
refFile    = fullfile(dataFolder, '参考.csv');  % 只有一行参考值
Tref = readtable(refFile);
ref_row = table2array(Tref(1, 2:end));  % 只取第一行数据（第2列开始）

% 4个文件（你可以自己改名字）
% fileNames = {
%     '90_YZ_1J.csv', ...
%     '90_YZ_1O.csv', ...
%     '90_YZ_2J.csv', ...
%     '90_YZ_2O.csv',...
%     '90_YZ_3J.csv', ...
%     '90_YZ_3O.csv', ...
%     '90_YZ_4J.csv', ...
%     '90_YZ_4O.csv'...
% };

fileNames = {
    % '90_1_众数.csv', ...
    % '90_2_众数.csv', ...
    % '80_1_众数.csv', ...
    % '80_2_众数.csv',...
    % '80_3_众数.csv', ...
    '80_1_众数_偶数.csv', ...
    '80_2_众数_偶数.csv', ...
    '80_3_众数_偶数.csv',...
    '80_1_众数_奇数.csv', ...
    '80_2_众数_奇数.csv', ...
    '80_3_众数_奇数.csv'...
};
% 颜色 + 线型 自动适配 4 条线
baseColors = lines(8);
fileColors = baseColors;
lineStyles = {'-', '--', ':', '-.','-', '--', ':', '-.'};  % 4种线型

% 初始化存储变量
numFiles = length(fileNames);
numVars = 16; 
Vertex_X = zeros(numFiles, numVars); 
Vertex_Y = zeros(numFiles, numVars);
SavedColNames = cell(1, numVars);

figure('Name', 'e_nose 4文件归一化趋势对比', 'Color', 'w');
t = tiledlayout(4, 4, 'TileSpacing', 'compact', 'Padding', 'compact'); % 布局
title(t, 'e_nose 4文件各列数据归一化趋势及顶点', 'FontSize', 14);

legendHandles = []; 

% ==========================================================
% 2. 循环每一列
% ==========================================================
for colIdx = 2:17
    varIdx = colIdx - 1;
    nexttile;
    hold on; grid on;
    currentColName = ''; 
    
    % ======================================================
    % 3. 循环每一个文件
    % ======================================================
    for fIdx = 1:numFiles
        
        % 拼接路径：e_nose/文件名.xlsx
        fname = fullfile(dataFolder, fileNames{fIdx});
        
        % 读取表格
        try
            T = readtable(fname);
        catch
            warning(['无法读取: ' fname]);
            continue;
        end
        
        % 取数据
        cdate = T{:, 1}; 
        target = 6;
        if colIdx ~= colIdx
            raw_pop = T{:, target} - T{:, colIdx};
        else
            raw_pop = T{:, colIdx};
        end
        
        % 记录列名
        if fIdx == 1
            currentColName = T.Properties.VariableNames{colIdx};
            SavedColNames{varIdx} = currentColName;
        end
        
        %% 归一化
        % MAX_MIN
        % min_val = min(raw_pop);
        % max_val = max(raw_pop);
        % if max_val == min_val
        %     pop_norm = raw_pop - min_val;
        % else
        %     pop_norm = (raw_pop - min_val) / (max_val - min_val);
        % end

        % Z_score
        % mu = mean(raw_pop);
        % sigma = std(raw_pop);
        % pop_norm = (raw_pop - mu) / sigma;
        % 均值归一化
        mu = mean(raw_pop);
        min_val = min(raw_pop);
        max_val = max(raw_pop);
        pop_norm = (raw_pop - mu) / (max_val - min_val);
        % 二次拟合（核心！）
        [f, ~] = fit(cdate, pop_norm, 'poly7');
        
        % 计算顶点
        x_vertex = -f.p2 / (2 * f.p1);
        y_vertex = f(x_vertex);
        
        % 保存顶点
        Vertex_X(fIdx, varIdx) = x_vertex;
        Vertex_Y(fIdx, varIdx) = y_vertex;
        
        % ======================================================
        % 绘图
        % ======================================================
        % % 画：未拟合的原始归一化曲线（真实数据）
        % plot(cdate, pop_norm, ...
        %     'Color', fileColors(fIdx, :), ...
        %     'LineStyle', 'none', ...
        %     'Marker', '.', ...
        %     'MarkerSize', 4, ...
        %     'HandleVisibility', 'off'); 
        
        % 画：拟合后的光滑曲线（你原来的） f(cdate)
        p = plot(cdate, pop_norm, ...
             'Color', fileColors(fIdx, :), ...
             'LineStyle', lineStyles{fIdx}, ...
             'LineWidth', 1, ...
             'DisplayName', fileNames{fIdx}); 
         
        if colIdx == 2 
            legendHandles = [legendHandles, p];
        end

        % 画顶点
        if x_vertex >= min(cdate) && x_vertex <= max(cdate)
            plot(x_vertex, y_vertex, 'p', ... 
                 'MarkerSize', 10, 'MarkerEdgeColor', 'k', ...
                 'MarkerFaceColor', fileColors(fIdx, :), ...
                 'HandleVisibility', 'off'); 
        end
    end
    
    title(currentColName, 'Interpreter', 'none'); 
    axis tight;
    if colIdx >= 5, xlabel('Cdate'); end
    if colIdx == 2 || colIdx == 5, ylabel('Norm. Value'); end
    
    hold off;
end

% ==========================================================
% 4. 共享图例
% ==========================================================
lgd = legend(legendHandles, 'Interpreter', 'none');
lgd.Layout.Tile = 'east'; 

% 保存结果
save('InflectionPoints_e_nose.mat', 'Vertex_X', 'Vertex_Y', 'fileNames', 'SavedColNames');