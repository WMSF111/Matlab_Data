% 1. 读取数据
% 建议使用 readtable 以便自动获取列名
T = readtable('90_PH_Mrix.xlsx');

% 提取 X 轴数据 (假设第1列)
% T{:, 1} 将 table 转为数值向量
cdate = T{:, 1}; 

% 2. 准备绘图
figure; 
hold on; % 保持绘图
axis tight; 
grid on;
title('归一化后的数据拟合对比 (0-1范围)');
xlabel('Cdate (X轴)');
ylabel('Normalized Value (归一化数值)');

% 用于存储图例名称和句柄
legend_str = {}; 
line_handles = [];

% 生成一组颜色，确保每条线颜色不同
colors = lines(6); 

% 3. 循环处理第2列到第7列
for i = 2:7
    % --- A. 提取数据 ---
    raw_pop = T{:, i}; % 原始 Y 数据
    
    % --- B. 归一化处理 (Min-Max Normalization) ---
    % 公式：(x - min) / (max - min)
    min_val = min(raw_pop);
    max_val = max(raw_pop);
    
    % 防止分母为0 (即整列数据都一样的情况)
    if max_val - min_val == 0
        pop_norm = raw_pop - min_val; % 全变0
    else
        pop_norm = (raw_pop - min_val) / (max_val - min_val);
    end
    
    % --- C. 拟合 (针对归一化后的数据) ---
    [f, gof] = fit(cdate, pop_norm, 'poly2');
    
    % --- D. 绘图 ---
    % 获取列名
    col_name = T.Properties.VariableNames{i};
    
    % 绘制拟合曲线
    % 使用 i-1 作为颜色索引，因为 i 从 2 开始
    h = plot(cdate, f(cdate), 'Color', colors(i-1,:), 'LineWidth', 2);
    
    % 收集信息用于图例
    line_handles = [line_handles, h];
    legend_str{end+1} = col_name;
end

% 4. 添加图例
legend(line_handles, legend_str, 'Interpreter', 'none', 'Location', 'best');
hold off;