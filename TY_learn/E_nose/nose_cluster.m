%% ========================
%% 电子鼻 自动特征组合聚类工具【15组超强版】
%% 自动组合 | 自动出图 | 自动输出簇内容 | 自动检查分类效果
%% ========================
clear; clc; close all;
set(0,'DefaultAxesFontName','SimHei');
set(0,'DefaultAxesFontName','SimHei');

%% 1. 路径
rootDirs = {
    '80\1_去基\奇数', ...
    '80\2_去基\奇数', ...
    '80\3_去基\奇数', ...
    '80\1_去基\偶数', ...
    '80\2_去基\偶数', ...
    '80\3_去基\偶数',...
    '70\1_去基\奇数', ...
    '70\2_去基\奇数', ...
    '70\3_去基\奇数', ...
    '70\1_去基\偶数', ...
    '70\2_去基\偶数', ...
    '70\3_去基\偶数'...
};
classNumbers = 0:20;

%% 2. 读取所有数据 + 提取你的全部特征（不动你的代码！）
allData = [];
allFilePaths = {};

for cls = classNumbers
    for d = 1:length(rootDirs)
        filePath = fullfile(rootDirs{d}, sprintf('%d.csv', cls));
        if ~exist(filePath,'file'), continue; end
        
        T = readtable(filePath);
        T(:,1) = [];
        data = table2array(T); % 读取这个文件的数据
        selectedChannels = [2,12]; % 选择传感器
        data_sub = data(:, selectedChannels);
        data_sub = data;
        
        
        % ======================
        % 全部特征
        % ======================
        f_mean  = mean(data_sub);                  % 均值
        f_max   = max(data_sub);                   % 峰值
        f_min   = min(data_sub);                   % 最小值
        f_auc   = sum(data_sub,1);                 % 曲线下面积
        f_sum   = sum(data_sub,1);        % 总和（整体强度）
        f_med   = median(data_sub);       % 中位数（稳健大小）
        f_mode  = mode(data_sub);         % 众数（你要的）

        f_std    = std(data_sub);                  % 波动大小 → 越抖值越大
        f_slope  = mean(diff(data_sub));            % 整体趋势 → 上升=正 / 下降=负
        f_cv     = std(data_sub)./(mean(data_sub)+eps); % 相对波动 → 波动/均值（和大小无关）
        % f_curv   = mean(diff(diff(data_sub)));      % 弯曲程度 → 上凸/下弯
        % f_kurt   = kurtosis(zscore(data_sub));      % 峰的尖锐度 → 越尖值越大
        % f_skew   = skewness(zscore(data_sub));      % 对称性 → 左偏/右偏

        % 保存所有特征
        featAll = [f_mean,f_max,f_min,f_auc, f_sum, f_med, f_mode, f_std,f_slope,f_cv];
        allData = [allData; featAll];
        allFilePaths{end+1} = filePath;
    end
end

% ================================
% 特征名字（和上面顺序严格对应）
% ================================
featNames = {
    'mean','max','min','auc','sum', 'med', 'mode', 'std','slope','cv'
};

%% ==================================================================
%% 特征组合 + 名称 【完全对应】
%% ==================================================================
featureGroups = {
    [1],[2],[3],[4],[5],[6],[7],[8],[9],[10],...
    [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],...
    [2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],[2,10],...
    [3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],...
    [4,5],[4,6],[4,7],[4,8],[4,9],[4,10],...
    [5,6],[5,7],[5,8],[5,9],[5,10],...
    [6,7],[6,8],[6,9],[6,10],...
    [7,8],[7,9],[7,10],...
    [8,9],[8,10],...
    [9,10],...
    [2,5,6],[2,5,10],[2,6,10],[5,6,10],[1,2,5],[1,2,6],[2,5,7],[2,6,7],...
    [2,5,6,10],[1,2,5,6],[2,5,6,7],[5,6,9,10],[1,2,4,5],...
    [2,5,6,9,10],[5,6,7,9,10],[2,5,6,7,10],[1,2,4,5,6],[2,5,6,8,10],...
    [1,2,4,5,6,10],[1,2,3,4,5,6],[5,6,7,8,9,10],...
    [2,5,6,7,8,9,10],[1,2,4,5,6,9,10],[1,2,4,5,6,7,10],...
    [1,2,4,5,6,7,9,10],[2,3,5,6,7,8,9,10],...
    [1,2,4,5,6,7,8,9,10],...
    [1,2,3,4,5,6,7,8,9,10]
};

% ================================
% ✅ 修复：groupNames 完全对应
% ================================
groupNames = {
    '均值','最大值','最小值','面积','总和','中位数','众数','标准差','斜率','变异系数',...
    '1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10',...
    '2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10',...
    '3-4','3-5','3-6','3-7','3-8','3-9','3-10',...
    '4-5','4-6','4-7','4-8','4-9','4-10',...
    '5-6','5-7','5-8','5-9','5-10',...
    '6-7','6-8','6-9','6-10',...
    '7-8','7-9','7-10',...
    '8-9','8-10',...
    '9-10',...
    '2+5+6','2+5+10','2+6+10','5+6+10','1+2+5','1+2+6','2+5+7','2+6+7',...
    '2+5+6+10','1+2+5+6','2+5+6+7','5+6+9+10','1+2+4+5',...
    '2+5+6+9+10','5+6+7+9+10','2+5+6+7+10','1+2+4+5+6','2+5+6+8+10',...
    '1+2+4+5+6+10','1+2+3+4+5+6','5+6+7+8+9+10',...
    '2+5+6+7+8+9+10','1+2+4+5+6+9+10','1+2+4+5+6+7+10',...
    '1+2+4+5+6+7+9+10','2+3+5+6+7+8+9+10',...
    '9特征全集','10特征终极全集'
};

num_clusters = 2;

%% ==================================================================
%% 遍历 + 聚类 + 输出
%% ==================================================================
for g = 1:length(featureGroups)
    featIdx = featureGroups{g}; 
    X = allData(:,featIdx);
    nDim = size(X,2);

    rng(1);
    if nDim == 1
        map = X;
        x_sorted = sort(X);
        th = x_sorted(round(length(X)/2));
        cluster_idx = ones(size(X,1),1);
        cluster_idx(X > th) = 2;
    else
        [coeff,score] = pca(X);
        map = score(:,1:2); 
        % map = tsne(X, 'NumDimensions', 2, 'Perplexity', 10);
        % cluster_idx = kmeans(map, num_clusters,'Replicates',50);
        D = pdist(map, 'cityblock');      % 曼哈顿距离
        Z = linkage(D, 'average');      % 层次聚类
        cluster_idx = cluster(Z, 'maxclust', num_clusters);
    end

    % 异常点
    try
        [~,score] = pca(X);
        mahal = sum(score.^2,2);
        is_outlier = mahal > mean(mahal)+4*std(mahal);
    catch
        is_outlier = false(size(X,1),1);
    end

    % ================= 计算 3 个簇的种类 =================
    idx1 = find(cluster_idx == 1);
    class1 = [];
    for i = 1:length(idx1)
        num = regexp(allFilePaths{idx1(i)},'(\d+)\.csv','tokens');
        class1 = [class1, str2double(num{1}{1})];
    end
    c1 = length(unique(class1));

    idx2 = find(cluster_idx == 2);
    class2 = [];
    for i = 1:length(idx2)
        num = regexp(allFilePaths{idx2(i)},'(\d+)\.csv','tokens');
        class2 = [class2, str2double(num{1}{1})];
    end
    c2 = length(unique(class2));

    idx3 = find(cluster_idx == 3);  % 👈 新增簇3
    class3 = [];
    for i = 1:length(idx3)
        num = regexp(allFilePaths{idx3(i)},'(\d+)\.csv','tokens');
        class3 = [class3, str2double(num{1}{1})];
    end
    c3 = length(unique(class3));

    totalType = c1 + c2 + c3;

    % ====================== 过滤条件（3 簇） ======================
    % if totalType < 27 && c1 >= 4 && c2 >= 4 && c3 >= 4 && g < 20  % 👈 3 簇条件
    if totalType < 24 && c1 >= 4 && c2 >= 4  % 👈 2 簇条件
        fprintf('\n\n===============================================================\n');
        fprintf(' 组合 %d → %s\n',g,groupNames{g});
        fprintf(' 特征：%s\n',strjoin(featNames(featIdx),', '));
        fprintf(' 簇1：%d种 | 簇2：%d种 | 簇3：%d种 | 总计：%d\n',c1,c2,c3,totalType);
        fprintf('===============================================================\n');
        
        % 输出 3 个簇
        for k = 1:num_clusters
            fprintf('\n📦 簇 %d\n',k);
            idx = find(cluster_idx == k);
            fileList = allFilePaths(idx);
            classInCluster = [];
            % for i = 1:length(idx)
            %     fname = allFilePaths{idx(i)};
            % 
            %     % 🔥 提取文件夹名称 70 / 80
            %     folderNum = regexp(fname, '[78]0', 'match');
            %     if ~isempty(folderNum)
            %         classInCluster = [classInCluster, str2double(folderNum{1})];
            %     end
            % 
            %     fprintf('  %s\n', fname);
            % end
            classInCluster = [];
            for i = 1:length(idx)
                num = regexp(fileList{i},'(\d+)\.csv','tokens');
                classInCluster = [classInCluster, str2double(num{1}{1})];
                fprintf('  %s\n', fileList{i});
            end
            classU = unique(classInCluster);
            fprintf(' 气体种类：'); fprintf('%d ',sort(classU)); fprintf('\n');
        end


        % 异常点
        fprintf('\n⚫ 异常点：');
        outFiles = allFilePaths(is_outlier);
         if isempty(outFiles)
            fprintf('  无\n');
        else
            for i=1:length(outFiles)
                fprintf('  %s\n', outFiles{i});
            end
         end
        % % ==========================================================
        % % 🔥 下面是：自动绘制聚类效果图（直接加上）
        % % ==========================================================
        % figure('Position',[100,100,800,600]);
        % hold on; grid on;
        % 
        % idx1 = cluster_idx == 1;
        % idx2 = cluster_idx == 2;
        % idx3 = cluster_idx == 3;
        % idxo = is_outlier;
        % 
        % % 画簇1
        % scatter(map(idx1,1), map(idx1,2), 80, 'b', 'filled', 'MarkerEdgeColor','k');
        % % 画簇2
        % scatter(map(idx2,1), map(idx2,2), 80, 'r', 'filled', 'MarkerEdgeColor','k');
        % % 画簇3
        % scatter(map(idx3,1), map(idx3,2), 80, 'green', 'filled', 'MarkerEdgeColor','k');
        % % 画异常点
        % if sum(idxo) > 0
        %     scatter(map(idxo,1), map(idxo,2), 120, 'k', 'x', 'LineWidth',2);
        % end
        % 
        % % 显示每个点的气体编号
        % for i = 1:size(map,1)
        %     fname = allFilePaths{i};
        %     num = regexp(fname,'(\d+)\.csv','tokens');
        %     gasNum = str2double(num{1}{1});
        %     text(map(i,1)+0.05, map(i,2)+0.05, sprintf('%d',gasNum),'FontSize',10);
        % end
        % 
        % title(sprintf('聚类效果 | 组合%d：%s',g,groupNames{g}),'FontSize',14);
        % xlabel('t-SNE Dimension 1','FontSize',12);
        % ylabel('t-SNE Dimension 2','FontSize',12);
        % legend('簇1','簇2','簇3','异常点','Location','best');
        % hold off;
        % saveas(gcf, sprintf('聚类图_组合%d.png',g)); % 如需自动保存图片，打开这行
    end
end
fprintf('\n🎉 全部运行完毕！\n');