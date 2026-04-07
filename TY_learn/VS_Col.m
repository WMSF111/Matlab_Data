clear;
clc;
% ==========================================================
% 1. 配置部分（适配子文件夹 + 4个文件）
% ==========================================================
% 请修改为你的实际子文件夹路径
% dataFolder = 'E_nose'; % 示例：'实验数据' 或 'D:\project\data'
% 
% % 文件列表
% fileNames = {'80_2_液体_众数.csv', '80_1_液体_众数.csv', '80_1_固体_众数.csv',...
%     '80_2_固体_众数.csv'}; 
% fileNames = {'80_PH_Mrix_1.xlsx', '80_PH_Mrix_2.xlsx', '80_PH_Mrix_3.xlsx',...
%     '80_PH_Mrix_1.xlsx', '80_PH_Mrix_2.xlsx', '80_PH_Mrix_3.xlsx',...
%     '80_PH_Mrix_2.xlsx', '80_PH_Mrix_3.xlsx'}; 
set(0,'DefaultAxesFontName','SimHei');  % 黑体
set(0,'DefaultTextFontName','SimHei');
set(0,'DefaultLegendFontName','SimHei');
dataFolder = 'NIR';  % 你的文件夹名称
fileNames = {'80_PH_Mrix_1_1.xlsx', '80_PH_Mrix_1_2.xlsx', '80_PH_Mrix_1_3.xlsx',...
    '80_PH_Mrix_2_1.xlsx', '80_PH_Mrix_2_2.xlsx', '80_PH_Mrix_2_3.xlsx',...
    '80_PH_Mrix_3_1.xlsx', '80_PH_Mrix_3_2.xlsx', '80_PH_Mrix_3_3.xlsx'}; 

baseColors = lines(3); 
fileColors = repelem(baseColors, 3, 1);
lineStyles = {'-', '--', ':', '-', '--', ':', '-', '--', ':'}; 

% 初始化存储变量
numFiles = length(fileNames);
numVars = 7; 
Vertex_X = zeros(numFiles, numVars); 
Vertex_Y = zeros(numFiles, numVars);
SavedColNames = cell(1, numVars);

figure('Name', '三个文件归一化趋势对比及极值点', 'Color', 'w');
% 【关键】tiledlayout 对象赋值给 t，后面要用来放图例
t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, '不同文件下各列数据归一化趋势及顶点标记', 'FontSize', 16);

% 【新增】用于临时存储图例需要的线条句柄
legendHandles = []; 

refFileName = '80_PH_Mrix_2_2.xlsx';

for colIdx = 2:numVars + 1
    varIdx = colIdx - 1;
    nexttile;
    hold on; grid on;
    currentColName = ''; 
    
    % ======================================================
    % 循环每个文件
    % ======================================================
    for fIdx = 1:numFiles
        
        fname = fullfile(dataFolder, fileNames{fIdx});
        
        % 安全读取表格
        try
            opts = detectImportOptions([fname]);
            opts.VariableNamingRule = 'preserve';  % 关键：禁止修改列名
            T = readtable(fname,opts);
        catch
            warning('读取失败：%s', fname);
            continue;
        end
        
        % 安全取数据（防止列不存在）
        if colIdx > width(T)
            warning('文件%s无第%d列，跳过', fname, colIdx);
            continue;
        end
        
        cdate = T{:, 1};
        raw_pop = T{:, colIdx};
        
        % 保存列名（只执行1次）
        if fIdx == 1
            currentColName = T.Properties.VariableNames{colIdx};
            SavedColNames{varIdx} = currentColName;
        end
        
        % 归一化
        pop_norm = normalizeData(raw_pop, 'maxmin');
        % 求斜率（微分）
        pop_norm = gradient(pop_norm, cdate); 
        
       % ========== ✅ 关键：poly9 + 中心化缩放（消病态警告）==========
        try
            % 加 'Normalize','on' 是核心！
            [f, ~] = fit(cdate, pop_norm, 'poly9', 'Normalize', 'on');
        catch ME
            warning('%s 拟合失败：%s', fname, ME.message);
            continue;
        end
        
        % ========== ✅ 兼容poly9：数值法找极值点（替代旧的a/b/c）==========
        x_fine = linspace(min(cdate), max(cdate), 1000);
        y_fit = f(x_fine);
        dy = diff(y_fit)./diff(x_fine);
        idx_cross = find(abs(dy(1:end-1).*dy(2:end)) < 1e-6 & dy(1:end-1).*dy(2:end) <= 0);
        
        if ~isempty(idx_cross)
            x_vertex = x_fine(idx_cross(1));
            y_vertex = f(x_vertex);
        else
            [y_vertex, idx_max] = max(y_fit);
            x_vertex = x_fine(idx_max);
        end
        
        Vertex_X(fIdx, varIdx) = x_vertex;
        Vertex_Y(fIdx, varIdx) = y_vertex;
        
        % ================= 绘图 =================
        p = plot(cdate,  f(cdate) , ...  % 这里修复！你原来写反了 f(cdate) 与 pop_norm
             'Color', fileColors(fIdx, :), ...
             'LineStyle', lineStyles{fIdx}, ...
             'LineWidth', 2, ...
             'DisplayName', fileNames{fIdx});  % 不要显示全路径，只显示文件名
    end
    
    title(currentColName, 'Interpreter', 'none'); 
    axis tight;
    if colIdx >= 5
        xlabel('Cdate');
    end
    if colIdx == 2 || colIdx == 5
        ylabel('Norm. Value');
    end
    
    hold off;
end

% ==========================================================
% 4. 生成共享图例 (放在最右侧)
% ==========================================================
% ==========================================================
% 【最终正确版】归一化后的数据计算 ΔE
% 先归一化 L*a*b* → 再减自己第一行 → 再算色差
% ==========================================================
fprintf('\n========== 绘制 0-20 归一化色差曲线 ==========\n');
fprintf('✅ 使用归一化后数据计算 ΔE\n');
fprintf('✅ 每个文件 - 自己第一行\n');
fprintf('✅ 自动跳过不存在的文件\n');

x_fixed = linspace(0, 20, 1000);

figure('Name','归一化后色差曲线 ΔE 0~20','Color','w');
hold on; grid on;
xlabel('Cdate');
ylabel('ΔE (Normalized)');
xlim([0, 20]);
title('归一化后色差曲线（自身基线校正）','FontSize',14);

for fIdx = 1:numFiles
    fname = fullfile(dataFolder, fileNames{fIdx});
    
    % 安全读取
    try
        opts = detectImportOptions(fname);
        opts.VariableNamingRule = 'preserve';
        T = readtable(fname, opts);
    catch
        warning('跳过：%s', fname);
        continue;
    end
    
    if width(T) < 8
        warning('列数不足，跳过：%s', fname);
        continue;
    end

    % ======================
    % 1. 取出原始 L*a*b*
    % ======================
    L_raw = T{:,5};
    a_raw = T{:,6};
    b_raw = T{:,7};

    % ======================
    % 2. 归一化（和主图一致：maxmin）
    % ======================
    L_norm = normalizeData(L_raw, 'maxmin');
    a_norm = normalizeData(a_raw, 'maxmin');
    b_norm = normalizeData(b_raw, 'maxmin');
    L_norm = gradient(L_norm, cdate); 
    a_norm = gradient(a_norm, cdate); 
    b_norm = gradient(b_norm, cdate); 

    % ======================
    % 3. 减去自己第一行（基线）
    % ======================
    L0 = L_norm(1);
    a0 = a_norm(1);
    b0 = b_norm(1);
    
    dL = L_norm - L0;
    da = a_norm - a0;
    db = b_norm - b0;

    % ======================
    % 4. 计算归一化后的色差
    % ======================
    dE = sqrt(dL.^2 + da.^2 + db.^2);

    % ======================
    % 5. 插值 + 绘图
    % ======================
    x = T{:,1};
    dE_interp = interp1(x, dE, x_fixed, 'linear', 'extrap');
    
    plot(x_fixed, dE_interp, ...
         'Color', fileColors(fIdx,:), ...
         'LineStyle', lineStyles{fIdx}, ...
         'LineWidth', 2, ...
         'DisplayName', fileNames{fIdx});
end

legend('Location','best');
hold off;
fprintf('✅ 归一化色差曲线绘制完成！\n');