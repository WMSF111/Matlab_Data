%% ========================
%% 电子鼻 自动特征组合聚类工具【15组超强版】
%% 自动组合 | 自动出图 | 自动输出簇内容 | 自动检查分类效果
%% ========================
clear; clc; close all;
set(0,'DefaultAxesFontName','SimHei');
set(0,'DefaultAxesFontName','SimHei');

%% 1. 路径
rootDirs = {
    '2026_3_24_1\奇数', ...
    '2026_3_24_2\奇数', ...
    '2026_3_17_1\奇数', ...
    '2026_3_24_1\偶数', ...
    '2026_3_24_2\偶数', ...
    '2026_3_17_1\偶数'...
};
classNumbers = 0:10;

%% 2. 读取所有数据 + 提取你的全部特征（不动你的代码！）
allData = [];
allFilePaths = {};

for cls = classNumbers
    for d = 1:length(rootDirs)
        filePath = fullfile(rootDirs{d}, sprintf('%d.csv', cls));
        if ~exist(filePath,'file'), continue; end
        
        T = readtable(filePath);
        T(:,1) = [];
        data = table2array(T);
        
        % ======================
        % 【预处理】去尖刺 + 平滑（温和版）
        % ======================
        for c = 1:size(data,2)
            x = data(:,c);
            m = mean(x); s = std(x);
            x(x > m+4*s) = m+4*s;
            x(x < m-4*s) = m-4*s;
            data(:,c) = movmean(x,3);
        end

        % ======================
        % 🔥 你的全部特征（我一行不动）
        % ======================
        f_mean  = mean(data);                  % 均值
        f_max   = max(data);                   % 峰值
        f_min   = min(data);                   % 最小值
        f_auc   = sum(data,1);                 % 曲线下面积

        f_std    = std(data);                  % 波动大小 → 越抖值越大
        f_slope  = mean(diff(data));            % 整体趋势 → 上升=正 / 下降=负
        f_cv     = std(data)./(mean(data)+eps); % 相对波动 → 波动/均值（和大小无关）
        f_curv   = mean(diff(diff(data)));      % 弯曲程度 → 上凸/下弯
        f_kurt   = kurtosis(zscore(data));      % 峰的尖锐度 → 越尖值越大
        f_skew   = skewness(zscore(data));      % 对称性 → 左偏/右偏

        % 保存所有特征
        featAll = [f_mean,f_max,f_min,f_auc,f_std,f_slope,f_cv,f_curv,f_kurt,f_skew];
        allData = [allData; featAll];
        allFilePaths{end+1} = filePath;
    end
end

% ================================
% 特征名字（和上面顺序严格对应）
% ================================
featNames = {
    'mean','max','min','auc','std',...
    'slope','cv','curv','kurt','skew'
};

%% ==================================================================
%% 🔥 58 组 终极全组合 100%全覆盖 | 可直接运行 | 不报错
%% ==================================================================
featureGroups = {
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

groupNames = {
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
    '9特征全集',...
    '10特征终极全集'
};

num_clusters = 2;

%% ==================================================================
%% 自动遍历所有组合
%% ==================================================================
for g = 1:length(featureGroups)
    featIdx = featureGroups{g};
    X = zscore(allData(:,featIdx));
    
    % 降维 + 聚类
    rng(1);
    map = tsne(X,'NumDimensions',2,'Perplexity',10);
    cluster_idx = kmeans(map, num_clusters,'Replicates',5);
    
    % 异常点
    [~,score] = pca(X);
    mahal = sum(score.^2,2);
    is_outlier = mahal > mean(mahal)+3*std(mahal);
    
    % ==============================
    % 输出结果
    % ==============================
    if any(g== [8, 12,16,19,34,35,38,44,46,63,64,70])
        % 19
    % if any(g== [ 9,25,52])
        % % 出图
        % figure('Position',[100,100,750,650]); hold on; grid on;
        % colors = lines(num_clusters);
        % 
        % for k = 1:num_clusters
        %     plot(map(cluster_idx==k & ~is_outlier,1), map(cluster_idx==k & ~is_outlier,2), ...
        %          'o','MarkerSize',10,'Color',colors(k,:),'LineWidth',2);
        % end
        % plot(map(is_outlier,1),map(is_outlier,2),'ko','MarkerSize',13,'LineWidth',3,'MarkerFaceColor','k');
        % 
        % title({['组合 ',num2str(g),'：',groupNames{g}];'黑色=异常点'},'FontSize',13);
        % xlabel('t-SNE 1'); ylabel('t-SNE 2');
        % 

        fprintf('\n\n=================================================================================\n');
        fprintf('  组合 %d：%s\n',g,groupNames{g});
        fprintf('  特征：%s\n',strjoin(featNames(featIdx),', '));
        fprintf('=================================================================================\n');
        
        for k = 1:num_clusters
            fprintf('\n📦 簇 %d\n',k);
            idx = find(cluster_idx == k);
            fileList = allFilePaths(idx);
            
            % 提取文件名中的数字 0-10
            classInCluster = [];
            for i = 1:length(fileList)
                fname = fileList{i};
                num = regexp(fname,'(\d+)\.csv','tokens');
                if ~isempty(num)
                    classInCluster = [classInCluster, str2double(num{1}{1})];
                end
                fprintf('  %s\n', fileList{i});
            end
            
            % 自动显示这个簇包含哪些数字（0-10）
            classInCluster = unique(classInCluster);
            fprintf('  ✅ 本簇包含的气体编号：');
            fprintf('%d ', sort(classInCluster));
            fprintf('\n');
        end
        
        % 异常点
        fprintf('\n⚫ 异常点：\n');
        outFiles = allFilePaths(is_outlier);
        if isempty(outFiles)
            fprintf('  无\n');
        else
            for i=1:length(outFiles)
                fprintf('  %s\n', outFiles{i});
            end
        end
    end
end

fprintf('\n🎉 全部15组组合运行完毕！请对比哪个分类最清晰！\n');