clear;
clc;
% ==========================================================
% 强制开启中文支持
% ==========================================================
set(0,'DefaultAxesFontName','SimHei');
set(0,'DefaultTextFontName','SimHei');
set(0,'DefaultLegendFontName','SimHei');

% ==========================================================
% 1. 配置（你原来的完全不变）
% ==========================================================
dataFolder = 'test';
refFile    = fullfile(dataFolder, '参考.csv');
Tref = readtable(refFile);
ref_row = table2array(Tref(1, 2:end));

fileNames = {
    '80_1_众数_偶数.csv', ...
    '80_2_众数_偶数.csv', ...
    '80_3_众数_偶数.csv',...
    '80_1_众数_奇数.csv', ...
    '80_2_众数_奇数.csv', ...
    '80_3_众数_奇数.csv'...
};

baseColors = lines(8);
fileColors = baseColors;
lineStyles = {'-', '--', ':', '-.','-', '--'};

numFiles = length(fileNames);
numVars = 16;

Turn_X = zeros(numFiles, numVars);
Turn_Y = zeros(numFiles, numVars);
SavedColNames = cell(1, numVars);

% ===============================
% 全局存储：所有行的聚类特征
% ===============================
allFeatures = [];
allSourceFile = [];
allRowIndex = [];

% ==========================================================
% 2. 你原来的 16张图 100% 还原！
% ==========================================================
figure('Name', 'e_nose 趋势对比+顶端拐点', 'Color', 'w');
t = tiledlayout(4, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'e_nose 16传感器曲线 - 顶端拐点', 'FontSize', 14);

legendHandles = [];

for colIdx = 2:17
    varIdx = colIdx - 1;
    nexttile;
    hold on; grid on;
    currentColName = '';

    for fIdx = 1:numFiles
        fname = fullfile(dataFolder, fileNames{fIdx});
        try
            T = readtable(fname);
        catch
            continue;
        end

        cdate  = T{:, 1};
        raw    = T{:, colIdx};

        % 你原来的归一化
        mu      = mean(raw);
        min_val = min(raw);
        max_val = max(raw);
        norm    = (raw - mu) / (max_val - min_val);

        % ============================
        % [HOT] 顶端拐点（你要的）
        % ============================
        d1 = gradient(norm);
        peakIdx = find(d1(1:end-1) > 0 & d1(2:end) <= 0, 1);
        
        if ~isempty(peakIdx)
            x_turn = cdate(peakIdx);
            y_turn = norm(peakIdx);
        else
            [y_turn, peakIdx] = max(norm);
            x_turn = cdate(peakIdx);
        end

        Turn_X(fIdx, varIdx) = x_turn;
        Turn_Y(fIdx, varIdx) = y_turn;

        % 绘图（完全不变）
        p = plot(cdate, norm, ...
            'Color', fileColors(fIdx,:), ...
            'LineStyle', lineStyles{fIdx}, ...
            'LineWidth', 1, ...
            'DisplayName', fileNames{fIdx});

        if colIdx == 2
            legendHandles = [legendHandles, p];
        end

        plot(x_turn, y_turn, 's', ...
            'MarkerSize', 8, ...
            'MarkerEdgeColor', 'k', ...
            'MarkerFaceColor', fileColors(fIdx,:), ...
            'HandleVisibility','off');

        if fIdx == 1
            currentColName = T.Properties.VariableNames{colIdx};
            SavedColNames{varIdx} = currentColName;
        end
    end

    title(currentColName, 'Interpreter','none');
    axis tight;
    if colIdx >= 5, xlabel('Cdate'); end
    if colIdx == 2 || colIdx ==5, ylabel('Norm. Value'); end
    hold off;
end

% 图例
lgd = legend(legendHandles, 'Interpreter','none');
lgd.Layout.Tile = 'east';

save('TurningPoints_e_nose.mat','Turn_X','Turn_Y','fileNames','SavedColNames');

% ==========================================================
% 3. [HOT] 新增：读取所有文件 → 对【每一行】聚类
% 功能完整保留，图不消失！
% ==========================================================
fprintf('\n正在提取所有行的顶端拐点特征...\n');
numSensors = 16;
allFeatures = [];
allSourceFile = [];
allRowIndex = [];

for fIdx = 1:numFiles
    fname = fullfile(dataFolder, fileNames{fIdx});
    T = readtable(fname);
    dataMat = table2array(T(:, 2:17));
    numRows = height(dataMat);
    
    Xmat = zeros(numRows, numSensors);
    Ymat = zeros(numRows, numSensors);
    
    for s = 1:numSensors
        for r = 1:numRows
            y = dataMat(r, :);
            d1 = gradient(y);
            idx = find(d1(1:end-1)>0 & d1(2:end)<=0, 1);
            if ~isempty(idx)
                x = idx; yv = y(idx);
            else
                [yv, idx] = max(y);
                x = idx;
            end
            Xmat(r,s) = x;
            Ymat(r,s) = yv;
        end
    end
    
    feat = [Xmat, Ymat];
    allFeatures = [allFeatures; feat];
    allSourceFile = [allSourceFile; repmat({fileNames{fIdx}}, numRows, 1)];
    allRowIndex = [allRowIndex; (1:numRows)'];
end

