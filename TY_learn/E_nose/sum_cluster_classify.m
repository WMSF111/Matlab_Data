%% ========================
%% 电子鼻 自动特征组合聚类工具【15组超强版】
%% 自动组合 | 自动出图 | 自动输出簇内容 | 自动检查分类效果
%% ========================
clear; clc; close all;
set(0,'DefaultAxesFontName','SimHei');
set(0,'DefaultAxesFontName','SimHei');

%% 1. 路径
rootDirs = {
    '90\1_去基\奇数', ...
    '90\2_去基\偶数', ...
    '90\1_去基\奇数', ...
    '90\2_去基\偶数', ...
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
% rootDirs = {
%     '90\1\奇数', ...
%     '90\2\偶数', ...
%     '90\1\奇数', ...
%     '90\2\偶数', ...
%     '80\1\奇数', ...
%     '80\2\奇数', ...
%     '80\3\奇数', ...
%     '80\1\偶数', ...
%     '80\2\偶数', ...
%     '80\3\偶数',...
%     '70\1\奇数', ...
%     '70\2\奇数', ...
%     '70\3\奇数', ...
%     '70\1\偶数', ...
%     '70\2\偶数', ...
%     '70\3\偶数'...
% };
classNumbers = 0:20;

%% 2. 读取所有数据 + 提取你的全部特征（不动你的代码！）
allData = [];
allFilePaths = {};
allLabels = [];
allTempLabels = [];

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
        allLabels = [allLabels; cls];
        tempLabel = regexp(rootDirs{d}, '^[789]0', 'match', 'once');
        if isempty(tempLabel)
            error('Cannot find 70/80/90 label in folder path: %s', rootDirs{d});
        end
        allTempLabels = [allTempLabels; str2double(tempLabel)];
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

% 算法选择:
% 可视化模式: 'none' -> 仅评估并绘制二维可分性
% 无监督学习: 'hierarchical', 'kmeans', 'gmm', 'spectral', 'dbscan'
% 有监督学习:   'svm', 'knn', 'rf', 'lda'
algorithm = 'none';
labelMode = 'temp';  % 'class' = 0-20, 'temp' = 70/80/90
supervisedAlgorithms = {'svm','knn','rf','lda'};
topKToPrint = 5; % 输出准确率前topK的组合
enableParamSearch = true;
enableDimSearch = true;

dimReductionModes = {'none', 'pca'};
pcaDimList = [2, 3, 5];

svmKernelList = {'linear','gaussian'};
svmBoxList = [0.1, 1, 10];
svmScaleList = [0.1, 1, 10];

knnNeighborList = [1, 3, 5, 7];
knnDistanceList = {'euclidean','cityblock','cosine'};
knnStandardizeList = [false, true];

rfTreeList = [50, 100, 200];
rfLeafList = [1, 5, 10];

ldaTypeList = {'linear','diaglinear','pseudoLinear'};
knn2dNeighbors = [1, 5];
knn2dDistances = {'euclidean', 'cityblock'};

supervisedResults = struct('groupIdx', {}, 'groupName', {}, ...
    'featureText', {}, 'accuracy', {}, 'kfold', {}, ...
    'algorithm', {}, 'labelName', {}, 'dimText', {}, 'paramText', {}, ...
    'misIdx', {}, 'misTrue', {}, 'misPred', {}, 'misFiles', {});
visualResults = struct('groupIdx', {}, 'groupName', {}, ...
    'featureText', {}, 'accuracy', {}, 'labelName', {}, ...
    'dimText', {}, 'map2d', {}, 'labels', {}, ...
    'predLabels', {}, 'misIdx', {}, 'paramText', {});

%% ==================================================================
%% 遍历 + 聚类 + 输出
%% ==================================================================
for g = 1:length(featureGroups)
    featIdx = featureGroups{g}; 
    X = allData(:,featIdx);
    Xn = zscore(X);
    Xn(~isfinite(Xn)) = 0;
    nDim = size(Xn,2);

    rng(1);
    if strcmpi(lower(algorithm), 'none')
        fprintf('\n[2D] Feature group %d/%d -> %s\n', ...
            g, length(featureGroups), groupNames{g});
        fprintf('[2D] Features: %s\n', strjoin(featNames(featIdx), ', '));
        drawnow;

        switch lower(labelMode)
            case 'class'
                Y = allLabels;
                labelName = '0-20';
            case 'temp'
                Y = allTempLabels;
                labelName = '70/80/90';
            otherwise
                error('Unknown labelMode: %s', labelMode);
        end

        uniqueLabels = unique(Y);
        labelCounts = arrayfun(@(y)sum(Y == y), uniqueLabels);
        kfold2d = min(5, min(labelCounts));
        if kfold2d < 2
            fprintf('[2D] Skip %s: not enough samples per class.\n', groupNames{g});
            drawnow;
            continue;
        end

        if nDim == 1
            map2d = [Xn, zeros(size(Xn,1),1)];
            dimText2d = 'raw-1d-to-2d';
        elseif nDim == 2
            map2d = Xn;
            dimText2d = 'raw-2d';
        else
            [~, score2d, ~, ~, explained2d] = pca(Xn);
            map2d = score2d(:,1:2);
            dimText2d = sprintf('pca(2 dims, %.2f%% var)', sum(explained2d(1:2)));
        end

        cvp2d = cvpartition(Y, 'KFold', kfold2d);
        model2d = fitcknn(map2d, Y, 'NumNeighbors', 1, ...
            'Distance', 'euclidean', 'Standardize', true, ...
            'CVPartition', cvp2d);
        pred2d = kfoldPredict(model2d);
        acc2d = mean(pred2d == Y);
        misIdx2d = find(pred2d ~= Y);

        visualResults(end+1).groupIdx = g;
        visualResults(end).groupName = groupNames{g};
        visualResults(end).featureText = strjoin(featNames(featIdx), ', ');
        visualResults(end).accuracy = acc2d;
        visualResults(end).labelName = labelName;
        visualResults(end).dimText = dimText2d;
        visualResults(end).map2d = map2d;
        visualResults(end).labels = Y;
        visualResults(end).predLabels = pred2d;
        visualResults(end).misIdx = misIdx2d;

        fprintf('[2D] accuracy=%.2f%% | %s\n', acc2d * 100, dimText2d);
        drawnow;
        continue;
    end

    if ismember(lower(algorithm), supervisedAlgorithms)
        fprintf('\n[Progress] Feature group %d/%d -> %s\n', ...
            g, length(featureGroups), groupNames{g});
        fprintf('[Progress] Features: %s\n', strjoin(featNames(featIdx), ', '));
        drawnow;

        switch lower(labelMode)
            case 'class'
                Y = allLabels;
                labelName = '0-20';
            case 'temp'
                Y = allTempLabels;
                labelName = '70/80/90';
            otherwise
                error('Unknown labelMode: %s', labelMode);
        end

        uniqueLabels = unique(Y);
        labelCounts = arrayfun(@(y)sum(Y == y), uniqueLabels);
        kfold = min(5, min(labelCounts));

        if kfold < 2
            fprintf('\nSkip %s: not enough samples per class for cross validation.\n', groupNames{g});
            drawnow;
            continue;
        end

        cvp = cvpartition(Y, 'KFold', kfold);
        bestAcc = -inf;
        bestPredLabels = [];
        bestDimText = '';
        bestParamText = '';
        dimModes = {'none'};
        if enableDimSearch && nDim > 1
            dimModes = dimReductionModes;
        end

        for dimModeIdx = 1:numel(dimModes)
            currentDimMode = lower(dimModes{dimModeIdx});
            fprintf('[Progress] Trying dim mode: %s\n', currentDimMode);
            drawnow;
            switch currentDimMode
                case 'none'
                    Xmodel = Xn;
                    currentDimText = 'none';
                    currentPcaDims = [];
                case 'pca'
                    [~, scoreModel, ~, ~, explained] = pca(Xn);
                    validDims = pcaDimList(pcaDimList <= size(Xn,2));
                    if isempty(validDims)
                        validDims = min(2, size(Xn,2));
                    end
                    validDims = unique(validDims);
                otherwise
                    error('Unknown dim reduction mode: %s', currentDimMode);
            end

            if strcmp(currentDimMode, 'pca')
                dimLoopValues = validDims;
            else
                dimLoopValues = 0;
            end

            for dimValueIdx = 1:numel(dimLoopValues)
                if strcmp(currentDimMode, 'pca')
                    currentPcaDims = dimLoopValues(dimValueIdx);
                    Xmodel = scoreModel(:,1:currentPcaDims);
                    explainedRatio = sum(explained(1:currentPcaDims));
                    currentDimText = sprintf('pca(%d dims, %.2f%% var)', ...
                        currentPcaDims, explainedRatio);
                end
                fprintf('[Progress] Current projection: %s\n', currentDimText);
                drawnow;

                switch lower(algorithm)
                    case 'svm'
                        for kernelIdx = 1:numel(svmKernelList)
                            for boxIdx = 1:numel(svmBoxList)
                                if strcmpi(svmKernelList{kernelIdx}, 'linear')
                                    model = fitcecoc(Xmodel, Y, 'CVPartition', cvp, ...
                                        'Learners', templateSVM( ...
                                        'KernelFunction', svmKernelList{kernelIdx}, ...
                                        'BoxConstraint', svmBoxList(boxIdx)));
                                    paramText = sprintf('kernel=%s, box=%.3g', ...
                                        svmKernelList{kernelIdx}, svmBoxList(boxIdx));
                                    predLabels = kfoldPredict(model);
                                    acc = mean(predLabels == Y);
                                    if acc > bestAcc
                                        bestAcc = acc;
                                        bestPredLabels = predLabels;
                                        bestDimText = currentDimText;
                                        bestParamText = paramText;
                                        fprintf('[Best] acc=%.2f%% | %s | %s\n', ...
                                            acc * 100, bestDimText, bestParamText);
                                        drawnow;
                                    end
                                else
                                    for scaleIdx = 1:numel(svmScaleList)
                                        model = fitcecoc(Xmodel, Y, 'CVPartition', cvp, ...
                                            'Learners', templateSVM( ...
                                            'KernelFunction', svmKernelList{kernelIdx}, ...
                                            'BoxConstraint', svmBoxList(boxIdx), ...
                                            'KernelScale', svmScaleList(scaleIdx)));
                                        paramText = sprintf('kernel=%s, box=%.3g, scale=%.3g', ...
                                            svmKernelList{kernelIdx}, svmBoxList(boxIdx), ...
                                            svmScaleList(scaleIdx));
                                        predLabels = kfoldPredict(model);
                                        acc = mean(predLabels == Y);
                                        if acc > bestAcc
                                            bestAcc = acc;
                                            bestPredLabels = predLabels;
                                            bestDimText = currentDimText;
                                            bestParamText = paramText;
                                            fprintf('[Best] acc=%.2f%% | %s | %s\n', ...
                                                acc * 100, bestDimText, bestParamText);
                                            drawnow;
                                        end
                                    end
                                end
                            end
                        end
                    case 'knn'
                        for nIdx = 1:numel(knnNeighborList)
                            for distIdx = 1:numel(knnDistanceList)
                                for stdIdx = 1:numel(knnStandardizeList)
                                    model = fitcknn(Xmodel, Y, ...
                                        'NumNeighbors', knnNeighborList(nIdx), ...
                                        'Distance', knnDistanceList{distIdx}, ...
                                        'Standardize', knnStandardizeList(stdIdx), ...
                                        'CVPartition', cvp);
                                    paramText = sprintf('k=%d, distance=%s, standardize=%d', ...
                                        knnNeighborList(nIdx), knnDistanceList{distIdx}, ...
                                        knnStandardizeList(stdIdx));
                                    predLabels = kfoldPredict(model);
                                    acc = mean(predLabels == Y);
                                    if acc > bestAcc
                                        bestAcc = acc;
                                        bestPredLabels = predLabels;
                                        bestDimText = currentDimText;
                                        bestParamText = paramText;
                                        fprintf('[Best] acc=%.2f%% | %s | %s\n', ...
                                            acc * 100, bestDimText, bestParamText);
                                        drawnow;
                                    end
                                end
                            end
                        end
                    case 'rf'
                        for treeIdx = 1:numel(rfTreeList)
                            for leafIdx = 1:numel(rfLeafList)
                                model = fitcensemble(Xmodel, Y, 'Method', 'Bag', ...
                                    'NumLearningCycles', rfTreeList(treeIdx), ...
                                    'Learners', templateTree('MinLeafSize', rfLeafList(leafIdx)), ...
                                    'CVPartition', cvp);
                                paramText = sprintf('trees=%d, minleaf=%d', ...
                                    rfTreeList(treeIdx), rfLeafList(leafIdx));
                                predLabels = kfoldPredict(model);
                                acc = mean(predLabels == Y);
                                if acc > bestAcc
                                    bestAcc = acc;
                                    bestPredLabels = predLabels;
                                    bestDimText = currentDimText;
                                    bestParamText = paramText;
                                    fprintf('[Best] acc=%.2f%% | %s | %s\n', ...
                                        acc * 100, bestDimText, bestParamText);
                                    drawnow;
                                end
                            end
                        end
                    case 'lda'
                        for typeIdx = 1:numel(ldaTypeList)
                            model = fitcdiscr(Xmodel, Y, ...
                                'DiscrimType', ldaTypeList{typeIdx}, ...
                                'CVPartition', cvp);
                            paramText = sprintf('type=%s', ldaTypeList{typeIdx});
                            predLabels = kfoldPredict(model);
                            acc = mean(predLabels == Y);
                            if acc > bestAcc
                                bestAcc = acc;
                                bestPredLabels = predLabels;
                                bestDimText = currentDimText;
                                bestParamText = paramText;
                                fprintf('[Best] acc=%.2f%% | %s | %s\n', ...
                                    acc * 100, bestDimText, bestParamText);
                                drawnow;
                            end
                        end
                end
            end
        end

        predLabels = bestPredLabels;
        acc = bestAcc;
        misIdx = find(predLabels ~= Y);
        supervisedResults(end+1).groupIdx = g;
        supervisedResults(end).groupName = groupNames{g};
        supervisedResults(end).featureText = strjoin(featNames(featIdx), ', ');
        supervisedResults(end).accuracy = acc;
        supervisedResults(end).kfold = kfold;
        supervisedResults(end).algorithm = algorithm;
        supervisedResults(end).labelName = labelName;
        supervisedResults(end).dimText = bestDimText;
        supervisedResults(end).paramText = bestParamText;
        supervisedResults(end).misIdx = misIdx;
        supervisedResults(end).misTrue = Y(misIdx);
        supervisedResults(end).misPred = predLabels(misIdx);
        supervisedResults(end).misFiles = allFilePaths(misIdx);
        fprintf('[Done] Group %d best accuracy: %.2f%% | %s | %s\n', ...
            g, acc * 100, bestDimText, bestParamText);
        drawnow;
        continue;
    end

    if nDim == 1
        map = Xn;
        x_sorted = sort(Xn);
        th = x_sorted(round(length(Xn)/2));
        cluster_idx = ones(size(Xn,1),1);
        cluster_idx(Xn > th) = 2;
    else
        [~,score] = pca(Xn);
        map = score(:,1:2); 
        % map = tsne(Xn, 'NumDimensions', 2, 'Perplexity', 10);
        % cluster_idx = kmeans(map, num_clusters,'Replicates',50);
        D = pdist(map, 'cityblock');      % 曼哈顿距离
        Z = linkage(D, 'average');      % 层次聚类
        cluster_idx = cluster(Z, 'maxclust', num_clusters);
    end

    if ~strcmpi(algorithm, 'hierarchical')
        switch lower(algorithm)
            case 'kmeans'
                cluster_idx = kmeans(Xn, num_clusters, 'Replicates', 50);
            case 'gmm'
                gm = fitgmdist(Xn, num_clusters, 'Replicates', 20, ...
                    'RegularizationValue', 1e-6);
                cluster_idx = cluster(gm, Xn);
            case 'spectral'
                cluster_idx = spectralcluster(Xn, num_clusters);
            case 'dbscan'
                cluster_idx = dbscan(Xn, 0.8, 3);
                noiseIdx = cluster_idx == -1;
                if any(noiseIdx)
                    if all(noiseIdx)
                        cluster_idx(noiseIdx) = 1;
                    else
                        cluster_idx(noiseIdx) = max(cluster_idx(~noiseIdx)) + 1;
                    end
                end
            otherwise
                error('Unknown algorithm: %s', algorithm);
        end
    end

    % 异常点
    try
        [~,score] = pca(Xn);
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

if ismember(lower(algorithm), supervisedAlgorithms)
    if isempty(supervisedResults)
        fprintf('\nNo supervised results were generated.\n');
    else
        allAcc = [supervisedResults.accuracy];
        [~, sortIdx] = sort(allAcc, 'descend');
        topN = min(topKToPrint, numel(sortIdx));

        fprintf('\n\n===============================================================\n');
        fprintf(' Top %d supervised results | Algorithm: %s | Label: %s\n', ...
            topN, supervisedResults(sortIdx(1)).algorithm, ...
            supervisedResults(sortIdx(1)).labelName);
        fprintf('===============================================================\n');

        for i = 1:topN
            result = supervisedResults(sortIdx(i));
            fprintf('\nRank %d | Feature group %d -> %s\n', ...
                i, result.groupIdx, result.groupName);
            fprintf('Features: %s\n', result.featureText);
            fprintf('Dim reduction: %s\n', result.dimText);
            fprintf('Best params: %s\n', result.paramText);
            fprintf('%d-fold CV accuracy: %.2f%%\n', ...
                result.kfold, result.accuracy * 100);
            if isempty(result.misIdx)
                fprintf('Misclassified samples: none\n');
            else
                fprintf('Misclassified samples: %d\n', numel(result.misIdx));
                for j = 1:numel(result.misIdx)
                    fprintf('  true=%g pred=%g | %s\n', ...
                        result.misTrue(j), result.misPred(j), result.misFiles{j});
                end
            end
        end
    end
end

if strcmpi(lower(algorithm), 'none')
    if isempty(visualResults)
        fprintf('\nNo 2D visualization results were generated.\n');
    else
        allAcc2d = [visualResults.accuracy];
        [~, sortIdx2d] = sort(allAcc2d, 'descend');
        topN2d = min(topKToPrint, numel(sortIdx2d));

        fprintf('\n\n===============================================================\n');
        fprintf(' Top %d 2D visualization results | Label: %s\n', ...
            topN2d, visualResults(sortIdx2d(1)).labelName);
        fprintf('===============================================================\n');

        for i = 1:topN2d
            result = visualResults(sortIdx2d(i));
            fprintf('\nRank %d | Feature group %d -> %s\n', ...
                i, result.groupIdx, result.groupName);
            fprintf('Features: %s\n', result.featureText);
            fprintf('2D projection: %s\n', result.dimText);
            fprintf('%d-fold 2D KNN accuracy: %.2f%%\n', ...
                min(5, min(arrayfun(@(y)sum(result.labels == y), unique(result.labels)))), ...
                result.accuracy * 100);
            fprintf('Misclassified samples: %d\n', numel(result.misIdx));

            fig = figure('Position',[100,100,900,650]);
            h = gscatter(result.map2d(:,1), result.map2d(:,2), result.labels);
            hold on;
            grid on;
            title(sprintf('Top %d | Group %d: %s | %.2f%%', ...
                i, result.groupIdx, result.groupName, result.accuracy * 100), ...
                'FontSize', 14);
            xlabel('Dimension 1', 'FontSize', 12);
            ylabel('Dimension 2', 'FontSize', 12);

            classLabels = unique(result.labels);
            for c = 1:numel(classLabels)
                pts = result.map2d(result.labels == classLabels(c), :);
                edgeColor = h(c).Color;
                if size(pts,1) >= 3
                    hullIdx = convhull(pts(:,1), pts(:,2));
                    patch(pts(hullIdx,1), pts(hullIdx,2), edgeColor, ...
                        'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
                        'HandleVisibility', 'off');
                    plot(pts(hullIdx,1), pts(hullIdx,2), '-', ...
                        'Color', edgeColor, 'LineWidth', 1.8);
                elseif size(pts,1) == 2
                    plot(pts(:,1), pts(:,2), '-', ...
                        'Color', edgeColor, 'LineWidth', 1.8);
                end
            end

            if ~isempty(result.misIdx)
                scatter(result.map2d(result.misIdx,1), result.map2d(result.misIdx,2), ...
                    120, 'k', 'x', 'LineWidth', 2, 'DisplayName', 'Misclassified');
            end

            for j = 1:size(result.map2d,1)
                text(result.map2d(j,1)+0.03, result.map2d(j,2)+0.03, ...
                    num2str(result.labels(j)), 'FontSize', 8);
            end

            hold off;
            saveas(fig, sprintf('top2d_rank%d_group%d.png', i, result.groupIdx));
        end
    end
end

fprintf('\n🎉 全部运行完毕！\n');
