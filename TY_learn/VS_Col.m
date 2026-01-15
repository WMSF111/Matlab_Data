% ==========================================================
% 1. 配置部分
% ==========================================================
fileNames = {'70_PH_Mrix_1.xlsx', '70_PH_Mrix_2.xlsx', '70_PH_Mrix_3.xlsx',...
    '80_PH_Mrix_1.xlsx', '80_PH_Mrix_2.xlsx', '80_PH_Mrix_3.xlsx',...
    '90_PH_Mrix_2.xlsx', '90_PH_Mrix_3.xlsx'}; 

baseColors = lines(3); 
fileColors = repelem(baseColors, 3, 1);
lineStyles = {'-', '--', ':', '-', '--', ':', '-', '--', ':'}; 

% 初始化存储变量
numFiles = length(fileNames);
numVars = 6; 
Vertex_X = zeros(numFiles, numVars); 
Vertex_Y = zeros(numFiles, numVars);
SavedColNames = cell(1, numVars);

figure('Name', '三个文件归一化趋势对比及极值点', 'Color', 'w');
% 【关键】tiledlayout 对象赋值给 t，后面要用来放图例
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, '不同文件下各列数据归一化趋势及顶点标记', 'FontSize', 16);

% 【新增】用于临时存储图例需要的线条句柄
legendHandles = []; 

% ==========================================================
% 2. 循环每一列 (变量维度)
% ==========================================================
for colIdx = 2:7
    varIdx = colIdx - 1;
    nexttile;
    hold on; grid on;
    currentColName = ''; 
    
    % ======================================================
    % 3. 循环每一个文件
    % ======================================================
    for fIdx = 1:length(fileNames)
        
        % ... (读取数据、归一化、拟合、计算顶点代码不变) ...
        % 为了节省篇幅，省略中间计算过程，直接跳到绘图
        % 假设 f, cdate, x_vertex, y_vertex 已经算好了
        
        % 此处为了演示，模拟一下读取和计算过程（实际请保留你原有的读取/计算代码）
        % -----------------------------------------------------------
        fname = fileNames{fIdx};
        try
             T = readtable(fname);
        catch
             continue;
        end
        cdate = T{:, 1}; raw_pop = T{:, colIdx};
        if fIdx == 1, currentColName = T.Properties.VariableNames{colIdx}; SavedColNames{varIdx} = currentColName; end
        min_val = min(raw_pop); max_val = max(raw_pop);
        if max_val==min_val, pop_norm=raw_pop-min_val; else, pop_norm=(raw_pop-min_val)/(max_val-min_val); end
        [f,~]=fit(cdate,pop_norm,'poly2');
        x_vertex = -f.p2/(2*f.p1); y_vertex = f(x_vertex);
        Vertex_X(fIdx, varIdx) = x_vertex; Vertex_Y(fIdx, varIdx) = y_vertex;
        % -----------------------------------------------------------

        % --- E. 绘图 ---
        p = plot(cdate, f(cdate), ...
             'Color', fileColors(fIdx, :), ...
             'LineStyle', lineStyles{fIdx}, ...
             'LineWidth', 2, ...
             'DisplayName', fname); 
         
        % 【关键步骤】只在处理第1个子图时，收集线条句柄
        % 这样我们只收集 8 条线用于生成图例，避免重复
        if colIdx == 2 
            legendHandles = [legendHandles, p];
        end

        % 绘制顶点 (不计入图例)
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
    
    % 【注意】这里删掉了循环内的 legend 代码！
    hold off;
end

% ==========================================================
% 4. 生成共享图例 (放在最右侧)
% ==========================================================
% 利用收集好的 legendHandles 生成图例
lgd = legend(legendHandles, 'Interpreter', 'none');

% 【核心设置】将图例移动到布局的右侧 (East)
lgd.Layout.Tile = 'east'; 
% 如果觉得右侧太挤，可以改成 'south' (底部)
% lgd.Layout.Tile = 'south'; 

% 保存数据
save('InflectionPoints.mat', 'Vertex_X', 'Vertex_Y', 'fileNames', 'SavedColNames');