%% ========================
%% 电子鼻 【完整时序聚类】60秒全部保留 [OK]
%% 不压缩成单值 | 时序直接聚类 | 加预处理 | 标黑异常
%% ========================
clear; clc; close all;
set(0,'DefaultAxesFontName','SimHei');
set(0,'DefaultAxesFontName','SimHei');

%% 1. 你的8个文件夹
rootDirs = {
    '90', ...
    '80', ...
    '70' ...
};

classNumbers = 0:10; 

%% 2. 读取【完整60秒时序】+ 温和预处理（去尖刺+平滑）
allFeatures = [];   % 每一行 = 一条完整60秒时序曲线
allFilePaths = {};

for cls = classNumbers
    for d = 1:length(rootDirs)
        filePath = fullfile(rootDirs{d}, sprintf('%d.csv', cls));
        if ~exist(filePath,'file'), continue; end
        
        T = readtable(filePath);
        T(:,1) = [];
        data = table2array(T);  % 每一列 = 60秒时序
        
        % ======================
        % [OK] 温和预处理（只去干扰，不改变时序形状）
        % ======================
        for c = 1:size(data,2)
            x = data(:,c);
            m = mean(x); s = std(x);
            x(x > m + 4*s) = m + 4*s;   % 去尖刺
            x(x < m - 4*s) = m - 4*s;
            data(:,c) = movmean(x, 3); % 轻平滑
        end
        
        % ======================
        % [TARGET] 关键：直接用【完整时序】作为特征（60个点全部保留）
        % 不压缩、不求均值、不求最大值！
        % ======================
        feat = data(:);  % 把时序拉直成一行 → 保留全部时间信息
        
        allFeatures = [allFeatures; feat'];
        allFilePaths{end+1} = filePath;
    end
end

%% 3. 时序数据标准化（必须做）
X = zscore(allFeatures);

%% 4. 异常点检测（标黑，不删除）
[coeff, score] = pca(X);
mahalanobis = sum(score.^2, 2); 
mu_m = mean(mahalanobis);
s_m  = std(mahalanobis);
is_outlier = mahalanobis > mu_m + 3*s_m;

%% 5. 降维 + 聚类（基于完整时序）
rng(1)
map = tsne(X, 'NumDimensions',2, 'Perplexity',10);
num_clusters = 2;
cluster_idx = kmeans(map, num_clusters, 'Replicates',5);

%% 6. 绘图（正常点彩色，异常点黑色）
figure('Position',[100,100,800,700]); hold on; grid on;
colors = lines(num_clusters);

for k = 1:num_clusters
    idx = cluster_idx == k & ~is_outlier;
    plot(map(idx,1), map(idx,2), ...
         'o','MarkerSize',10,'LineWidth',2,'Color',colors(k,:),'DisplayName',['簇 ' num2str(k)]);
end

% 异常点 → 黑色
plot(map(is_outlier,1), map(is_outlier,2), ...
     'ko','MarkerSize',12,'LineWidth',3,'MarkerFaceColor','k','DisplayName','异常点');

title('【完整时序聚类】60秒全部保留（异常点=黑色）','FontSize',14);
xlabel('t-SNE 1'); ylabel('t-SNE 2');
legend('Location','best');

%% 7. 输出每簇包含的文件
fprintf('\n=========================================\n');
fprintf('              时序聚类结果\n');
fprintf('=========================================\n');
for k = 1:num_clusters
    fprintf('\n【簇 %d】:\n',k);
    idx = find(cluster_idx==k);
    for i=1:length(idx)
        fprintf('  %s\n', allFilePaths{idx(i)});
    end
end

%% 8. 输出异常点
fprintf('\n=========================================\n');
fprintf('             异常样本（标黑）\n');
fprintf('=========================================\n');
outlier_files = allFilePaths(is_outlier);
if isempty(outlier_files)
    fprintf('[OK] 无异常点\n');
else
    for i=1:length(outlier_files)
        fprintf('[WARN]  %s\n', outlier_files{i});
    end
end

fprintf('\n[DONE] 完成！使用【完整60秒时序】聚类！\n');