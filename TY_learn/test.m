clear;
clc;

% ==========================================================
% 中文显示
% ==========================================================
set(0,'DefaultAxesFontName','SimHei');
set(0,'DefaultTextFontName','SimHei');
set(0,'DefaultLegendFontName','SimHei');

% ==========================================================
% 1. 基础配置
% ==========================================================
dataFolder = 'E_nose\test';
fileNames = {
    '90_YZ_1J.csv','90_YZ_1O.csv','90_YZ_2J.csv','90_YZ_2O.csv',...
    '90_YZ_3J.csv','90_YZ_3O.csv','90_YZ_4J.csv','90_YZ_4O.csv',...
};

baseColors = lines(8);
numFiles = length(fileNames);
numVars  = 16;

Vertex_X = zeros(numFiles, numVars);
Vertex_Y = zeros(numFiles, numVars);

% ==========================================================
% 画图窗口 4×4
% ==========================================================
figure('Color','w');
t = tiledlayout(4,4,'TileSpacing','compact','Padding','compact');
title(t,'传感器原始数据 → 标准化 → 平滑曲线','FontSize',14);
legendHandles = [];

% ==========================================================
% 循环 16 个传感器
% ==========================================================
for colIdx = 2:17
    varIdx = colIdx - 1;
    nexttile; hold on; grid on;
    
    for fIdx = 1:numFiles
        fname = fullfile(dataFolder, fileNames{fIdx});
        T = readtable(fname);
        cdate = T{:,1};
        raw_pop = T{:,colIdx};  

        % Z-score 标准化
        mu = mean(raw_pop);
        sigma = std(raw_pop);
        pop_norm = (raw_pop - mu) / sigma;

        % 拟合
        [f, ~] = fit(cdate, pop_norm, 'smoothingspline');

        % 画图
        p = plot(cdate,pop_norm, 'Color', baseColors(fIdx,:), 'LineWidth', 1.2, 'DisplayName', fileNames{fIdx});
        
        if colIdx == 2
            legendHandles = [legendHandles, p];
        end

        % 平稳值
        x_stable = max(cdate);
        y_stable = f(x_stable);
        
        Vertex_X(fIdx, varIdx) = x_stable;
        Vertex_Y(fIdx, varIdx) = y_stable;
    end
    
    title(T.Properties.VariableNames{colIdx}, 'FontSize', 9);
end

% 图例
lgd = legend(legendHandles, 'Location', 'eastoutside');
lgd.Layout.Tile = 'east';

% ==========================================================
% 🚀 PCA 分析
% ==========================================================
X = [];
for f = 1:numFiles
    vec = Vertex_Y(f, :);
    X = [X; vec];
end
X = zscore(X);
[coeff, score, ~, ~, explained] = pca(X);

% ------------------------------
% PCA 图 1：散点图（原来的）
% ------------------------------
figure('Color','w','Position',[200,200,700,600]);
hold on; grid on;
for i = 1:numFiles
    plot(score(i,1), score(i,2), 'o', 'MarkerSize',10, 'LineWidth',2, 'Color',baseColors(i,:));
end
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PCA 得分图（散点）','FontSize',14);
legend(fileNames, 'Location','best');

% ------------------------------
% PCA 图 2：线型图（你要新增的）
% ------------------------------
figure('Color','w','Position',[300,300,700,600]);
hold on; grid on;

% 画线
plot(score(:,1), score(:,2), 'k-o', 'LineWidth',2.5, 'MarkerSize',10);

% 再上色点
for i = 1:numFiles
    plot(score(i,1), score(i,2), 'o', 'MarkerSize',10, 'Color',baseColors(i,:));
end

xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PCA 趋势线型图（按文件顺序连线）','FontSize',14);
legend(fileNames, 'Location','best');

% ==========================================================
% 保存
% ==========================================================
save('result.mat','Vertex_X','Vertex_Y','fileNames');
fprintf('✅ 全部完成！\n');