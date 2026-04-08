%% ========================
%% 鐢靛瓙榧?鑷姩鐗瑰緛缁勫悎鑱氱被宸ュ叿銆?5缁勮秴寮虹増銆?%% 鑷姩缁勫悎 | 鑷姩鍑哄浘 | 鑷姩杈撳嚭绨囧唴瀹?| 鑷姩妫€鏌ュ垎绫绘晥鏋?%% ========================
clear; clc; close all;
set(0,'DefaultAxesFontName','SimHei');
set(0,'DefaultAxesFontName','SimHei');

%% 1. 璺緞
rootDirs = {
    '90\1_鍘诲熀\濂囨暟', ...
    '90\2_鍘诲熀\鍋舵暟', ...
    '90\1_鍘诲熀\濂囨暟', ...
    '90\2_鍘诲熀\鍋舵暟', ...
    '80\1_鍘诲熀\濂囨暟', ...
    '80\2_鍘诲熀\濂囨暟', ...
    '80\3_鍘诲熀\濂囨暟', ...
    '80\1_鍘诲熀\鍋舵暟', ...
    '80\2_鍘诲熀\鍋舵暟', ...
    '80\3_鍘诲熀\鍋舵暟',...
    '70\1_鍘诲熀\濂囨暟', ...
    '70\2_鍘诲熀\濂囨暟', ...
    '70\3_鍘诲熀\濂囨暟', ...
    '70\1_鍘诲熀\鍋舵暟', ...
    '70\2_鍘诲熀\鍋舵暟', ...
    '70\3_鍘诲熀\鍋舵暟'...
};
% rootDirs = {
%     '90\1\濂囨暟', ...
%     '90\2\鍋舵暟', ...
%     '90\1\濂囨暟', ...
%     '90\2\鍋舵暟', ...
%     '80\1\濂囨暟', ...
%     '80\2\濂囨暟', ...
%     '80\3\濂囨暟', ...
%     '80\1\鍋舵暟', ...
%     '80\2\鍋舵暟', ...
%     '80\3\鍋舵暟',...
%     '70\1\濂囨暟', ...
%     '70\2\濂囨暟', ...
%     '70\3\濂囨暟', ...
%     '70\1\鍋舵暟', ...
%     '70\2\鍋舵暟', ...
%     '70\3\鍋舵暟'...
% };
classNumbers = 0:20;

%% 2. 璇诲彇鎵€鏈夋暟鎹?+ 鎻愬彇浣犵殑鍏ㄩ儴鐗瑰緛锛堜笉鍔ㄤ綘鐨勪唬鐮侊紒锛?allData = [];
allFilePaths = {};
allLabels = [];
allTempLabels = [];

for cls = classNumbers
    for d = 1:length(rootDirs)
        filePath = fullfile(rootDirs{d}, sprintf('%d.csv', cls));
        if ~exist(filePath,'file'), continue; end
        
        T = readtable(filePath);
        T(:,1) = [];
        data = table2array(T);
        selectedChannels = [2,12];
        data_sub = data(:, selectedChannels);
        data_sub = data;

        % ======================
        % 全部特征
        % ======================
        f_mean  = mean(data_sub);
        f_max   = max(data_sub);
        f_min   = min(data_sub);
        f_auc   = sum(data_sub,1);
        f_sum   = sum(data_sub,1);
        f_med   = median(data_sub);
        f_mode  = mode(data_sub);
        f_std   = std(data_sub);
        f_slope = mean(diff(data_sub));
        f_cv    = std(data_sub)./(mean(data_sub)+eps);

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
% 鐗瑰緛鍚嶅瓧锛堝拰涓婇潰椤哄簭涓ユ牸瀵瑰簲锛?% ================================
featNames = {
    'mean','max','min','auc','sum', 'med', 'mode', 'std','slope','cv'
};

%% ==================================================================
%% 鐗瑰緛缁勫悎 + 鍚嶇О 銆愬畬鍏ㄥ搴斻€?%% ==================================================================
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
% 鉁?淇锛歡roupNames 瀹屽叏瀵瑰簲
% ================================
groupNames = {
    '鍧囧€?,'鏈€澶у€?,'鏈€灏忓€?,'闈㈢Н','鎬诲拰','涓綅鏁?,'浼楁暟','鏍囧噯宸?,'鏂滅巼','鍙樺紓绯绘暟',...
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
    '9鐗瑰緛鍏ㄩ泦','10鐗瑰緛缁堟瀬鍏ㄩ泦'
};

num_clusters = 3;

% 绠楁硶閫夋嫨:
% 鍙鍖栨ā寮? 'none' -> 浠呰瘎浼板苟缁樺埗浜岀淮鍙垎鎬?% 鏃犵洃鐫ｅ涔? 'hierarchical', 'kmeans', 'gmm', 'spectral', 'dbscan'
% 鏈夌洃鐫ｅ涔?   'svm', 'knn', 'rf', 'lda'
algorithm = 'kmeans';
labelMode = 'temp';  % 'class' = 0-20, 'temp' = 70/80/90
supervisedAlgorithms = {'svm','knn','rf','lda'};
topKToPrint = 5; % 杈撳嚭鍑嗙‘鐜囧墠topK鐨勭粍鍚?enableParamSearch = true;
enableDimSearch = true;

dimReductionModes = {'none', 'pca', 'svd', 'mds', 'tsne'};
pcaDimList = [2, 3, 5];
tsnePerplexityList = [10, 20, 30];
mdsDistance = 'euclidean';
dimConfig = struct( ...
    'modes', {dimReductionModes}, ...
    'pcaDims', pcaDimList, ...
    'tsnePerplexities', tsnePerplexityList, ...
    'mdsDistance', mdsDistance);

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
unsupervisedResults = struct('groupIdx', {}, 'groupName', {}, ...
    'featureText', {}, 'algorithm', {}, 'labelName', {}, ...
    'dimText', {}, 'silhouette', {}, 'clusterIdx', {});
visualResults = struct('groupIdx', {}, 'groupName', {}, ...
    'featureText', {}, 'accuracy', {}, 'labelName', {}, ...
    'dimText', {}, 'map2d', {}, 'labels', {}, ...
    'predLabels', {}, 'misIdx', {}, 'paramText', {});

%% ==================================================================
%% 閬嶅巻 + 鑱氱被 + 杈撳嚭
%% ==================================================================
for g = 1:length(featureGroups)
    featIdx = featureGroups{g}; 
    [X, Xn, nDim] = prepare_feature_matrix(allData, featIdx);
    rng(1);
    [Y, labelName] = get_labels_by_mode(labelMode, allLabels, allTempLabels);

    if strcmpi(lower(algorithm), 'none')
        visualResult = evaluate_visual_mode(Xn, Y, featIdx, g, groupNames, ...
            featNames, labelName, knn2dNeighbors, knn2dDistances);
        if ~isempty(visualResult)
            visualResults(end+1) = visualResult;
        end
        continue;
    end

    if ismember(lower(algorithm), supervisedAlgorithms)
        supervisedResult = evaluate_supervised_mode(Xn, Y, featIdx, g, ...
            groupNames, featNames, labelName, algorithm, allFilePaths, ...
            enableDimSearch, dimConfig, ...
            svmKernelList, svmBoxList, svmScaleList, ...
            knnNeighborList, knnDistanceList, knnStandardizeList, ...
            rfTreeList, rfLeafList, ldaTypeList);
        if ~isempty(supervisedResult)
            supervisedResults(end+1) = supervisedResult;
        end
        continue;
    end

    [cluster_idx, map, unsupDimText, unsupScore] = evaluate_unsupervised_mode( ...
        Xn, algorithm, num_clusters, dimConfig);
    fprintf('[Unsupervised] best dim mode: %s | silhouette=%.4f\n', ...
        unsupDimText, unsupScore);
    unsupervisedResults(end+1) = struct( ...
        'groupIdx', g, ...
        'groupName', groupNames{g}, ...
        'featureText', strjoin(featNames(featIdx), ', '), ...
        'algorithm', algorithm, ...
        'labelName', labelName, ...
        'dimText', unsupDimText, ...
        'silhouette', unsupScore, ...
        'clusterIdx', cluster_idx);
    % 异常点
    try
        [~,score] = pca(Xn);
        mahal = sum(score.^2,2);
        is_outlier = mahal > mean(mahal)+4*std(mahal);
    catch
        is_outlier = false(size(X,1),1);
    end

    idx1 = find(cluster_idx == 1);
    class1 = Y(idx1)';
    c1 = length(unique(class1));

    idx2 = find(cluster_idx == 2);
    class2 = Y(idx2)';
    c2 = length(unique(class2));

    idx3 = find(cluster_idx == 3);
    class3 = Y(idx3)';
    c3 = length(unique(class3));

    totalType = c1 + c2 + c3;

    if totalType <= 7 && c1 <= 2 && c2 <= 2 && c3 <= 2
        fprintf('\n\n===============================================================\n');
        fprintf(' 组合 %d -> %s\n', g, groupNames{g});
        fprintf(' 特征: %s\n', strjoin(featNames(featIdx), ', '));
        fprintf(' 簇1: %d类 | 簇2: %d类 | 簇3: %d类 | 总计: %d\n', c1, c2, c3, totalType);
        fprintf('===============================================================\n');

        for k = 1:num_clusters
            fprintf('\n簇 %d\n', k);
            idx = find(cluster_idx == k);
            fileList = allFilePaths(idx);
            classInCluster = Y(idx)';
            for i = 1:length(idx)
                fprintf('  %s\n', fileList{i});
            end
            classU = unique(classInCluster);
            fprintf(' 类别: ');
            fprintf('%d ', sort(classU));
            fprintf('\n');
        end

        fprintf('\n异常点：');
        outFiles = allFilePaths(is_outlier);
        if isempty(outFiles)
            fprintf('  无\n');
        else
            for i = 1:length(outFiles)
                fprintf('  %s\n', outFiles{i});
            end
        end
    end
        % idx2 = cluster_idx == 2;
        % idx3 = cluster_idx == 3;
        % idxo = is_outlier;
        % 
        % % 鐢荤皣1
        % scatter(map(idx1,1), map(idx1,2), 80, 'b', 'filled', 'MarkerEdgeColor','k');
        % % 鐢荤皣2
        % scatter(map(idx2,1), map(idx2,2), 80, 'r', 'filled', 'MarkerEdgeColor','k');
        % % 鐢荤皣3
        % scatter(map(idx3,1), map(idx3,2), 80, 'green', 'filled', 'MarkerEdgeColor','k');
        % % 鐢诲紓甯哥偣
        % if sum(idxo) > 0
        %     scatter(map(idxo,1), map(idxo,2), 120, 'k', 'x', 'LineWidth',2);
        % end
        % 
        % % 鏄剧ず姣忎釜鐐圭殑姘斾綋缂栧彿
        % for i = 1:size(map,1)
        %     fname = allFilePaths{i};
        %     num = regexp(fname,'(\d+)\.csv','tokens');
        %     gasNum = str2double(num{1}{1});
        %     text(map(i,1)+0.05, map(i,2)+0.05, sprintf('%d',gasNum),'FontSize',10);
        % end
        % 
        % title(sprintf('鑱氱被鏁堟灉 | 缁勫悎%d锛?s',g,groupNames{g}),'FontSize',14);
        % xlabel('t-SNE Dimension 1','FontSize',12);
        % ylabel('t-SNE Dimension 2','FontSize',12);
        % legend('绨?','绨?','绨?','寮傚父鐐?,'Location','best');
        % hold off;
        % saveas(gcf, sprintf('鑱氱被鍥綺缁勫悎%d.png',g)); % 濡傞渶鑷姩淇濆瓨鍥剧墖锛屾墦寮€杩欒
    end
end

if ismember(lower(algorithm), supervisedAlgorithms)
    print_top_supervised_results(supervisedResults, topKToPrint);
end

if strcmpi(lower(algorithm), 'none')
    print_top_visual_results(visualResults, topKToPrint);
end

if ~strcmpi(lower(algorithm), 'none') && ~ismember(lower(algorithm), supervisedAlgorithms)
    print_top_unsupervised_results(unsupervisedResults, topKToPrint);
end

fprintf('\n鍏ㄩ儴杩愯瀹屾瘯锛乗n');

function [X, Xn, nDim] = prepare_feature_matrix(allData, featIdx)
X = allData(:, featIdx);
Xn = zscore(X);
Xn(~isfinite(Xn)) = 0;
nDim = size(Xn, 2);
end

function [Y, labelName] = get_labels_by_mode(labelMode, allLabels, allTempLabels)
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
end

function result = evaluate_visual_mode(Xn, Y, featIdx, g, groupNames, featNames, ...
    labelName, knn2dNeighbors, knn2dDistances)
result = struct();

fprintf('\n[2D] Feature group %d -> %s\n', g, groupNames{g});
fprintf('[2D] Features: %s\n', strjoin(featNames(featIdx), ', '));
drawnow;

uniqueLabels = unique(Y);
labelCounts = arrayfun(@(y)sum(Y == y), uniqueLabels);
kfold2d = min(5, min(labelCounts));
if kfold2d < 2
    fprintf('[2D] Skip %s: not enough samples per class.\n', groupNames{g});
    drawnow;
    return;
end

mapCandidates = struct('map2d', {}, 'dimText', {});
nDim = size(Xn, 2);
if nDim == 1
    mapCandidates(end+1).map2d = [Xn, zeros(size(Xn,1),1)];
    mapCandidates(end).dimText = 'raw-1d-to-2d';
else
    visualDimConfig = struct('modes', {{'pca'}}, 'pcaDims', 2, ...
        'tsnePerplexities', [], 'mdsDistance', 'euclidean');
    mapCandidates = build_dimensionality_candidates(Xn, visualDimConfig, 2);
    if nDim == 2
        rawCandidate.map2d = Xn;
        rawCandidate.dimText = 'raw-2d';
        mapCandidates = [rawCandidate, mapCandidates];
    end
end

cvp2d = cvpartition(Y, 'KFold', kfold2d);
bestAcc2d = -inf;
bestMap2d = [];
bestDimText2d = '';
bestPred2d = [];
bestMisIdx2d = [];
bestParam2d = '';

for mapIdx = 1:numel(mapCandidates)
    map2d = mapCandidates(mapIdx).map2d;
    dimText2d = mapCandidates(mapIdx).dimText;
    fprintf('[2D] Trying projection: %s\n', dimText2d);
    drawnow;

    for kIdx = 1:numel(knn2dNeighbors)
        for distIdx = 1:numel(knn2dDistances)
            paramText2d = sprintf('k=%d, distance=%s, standardize=1', ...
                knn2dNeighbors(kIdx), knn2dDistances{distIdx});
            model2d = fitcknn(map2d, Y, ...
                'NumNeighbors', knn2dNeighbors(kIdx), ...
                'Distance', knn2dDistances{distIdx}, ...
                'Standardize', true, ...
                'CVPartition', cvp2d);
            pred2d = kfoldPredict(model2d);
            acc2d = mean(pred2d == Y);
            if acc2d > bestAcc2d
                bestAcc2d = acc2d;
                bestMap2d = map2d;
                bestDimText2d = dimText2d;
                bestPred2d = pred2d;
                bestMisIdx2d = find(pred2d ~= Y);
                bestParam2d = paramText2d;
                fprintf('[2D-Best] acc=%.2f%% | %s | %s\n', ...
                    acc2d * 100, bestDimText2d, bestParam2d);
                drawnow;
            end
        end
    end
end

result.groupIdx = g;
result.groupName = groupNames{g};
result.featureText = strjoin(featNames(featIdx), ', ');
result.accuracy = bestAcc2d;
result.labelName = labelName;
result.dimText = bestDimText2d;
result.map2d = bestMap2d;
result.labels = Y;
result.predLabels = bestPred2d;
result.misIdx = bestMisIdx2d;
result.paramText = bestParam2d;

fprintf('[2D] best accuracy=%.2f%% | %s | %s\n', ...
    bestAcc2d * 100, bestDimText2d, bestParam2d);
drawnow;
end

function result = evaluate_supervised_mode(Xn, Y, featIdx, g, groupNames, featNames, ...
    labelName, algorithm, allFilePaths, enableDimSearch, dimConfig, ...
    svmKernelList, svmBoxList, svmScaleList, knnNeighborList, knnDistanceList, ...
    knnStandardizeList, rfTreeList, rfLeafList, ldaTypeList)
result = struct();

fprintf('\n[Progress] Feature group %d -> %s\n', g, groupNames{g});
fprintf('[Progress] Features: %s\n', strjoin(featNames(featIdx), ', '));
drawnow;

uniqueLabels = unique(Y);
labelCounts = arrayfun(@(y)sum(Y == y), uniqueLabels);
kfold = min(5, min(labelCounts));
if kfold < 2
    fprintf('\nSkip %s: not enough samples per class for cross validation.\n', groupNames{g});
    drawnow;
    return;
end

cvp = cvpartition(Y, 'KFold', kfold);
bestAcc = -inf;
bestPredLabels = [];
bestDimText = '';
bestParamText = '';

if enableDimSearch && size(Xn,2) > 1
    modelCandidates = build_dimensionality_candidates(Xn, dimConfig, max(2, size(Xn,2)));
else
    modelCandidates = build_dimensionality_candidates(Xn, dimConfig, max(2, size(Xn,2)), {'none'});
end
if isempty(modelCandidates)
    modelCandidates = build_dimensionality_candidates(Xn, dimConfig, max(2, size(Xn,2)), {'none'});
end

for projIdx = 1:numel(modelCandidates)
        Xmodel = modelCandidates(projIdx).map2d;
        currentDimText = modelCandidates(projIdx).dimText;
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
                            [bestAcc, bestPredLabels, bestDimText, bestParamText] = ...
                                update_best_result(model, Y, currentDimText, paramText, ...
                                bestAcc, bestPredLabels, bestDimText, bestParamText);
                        else
                            for scaleIdx = 1:numel(svmScaleList)
                                model = fitcecoc(Xmodel, Y, 'CVPartition', cvp, ...
                                    'Learners', templateSVM( ...
                                    'KernelFunction', svmKernelList{kernelIdx}, ...
                                    'BoxConstraint', svmBoxList(boxIdx), ...
                                    'KernelScale', svmScaleList(scaleIdx)));
                                paramText = sprintf('kernel=%s, box=%.3g, scale=%.3g', ...
                                    svmKernelList{kernelIdx}, svmBoxList(boxIdx), svmScaleList(scaleIdx));
                                [bestAcc, bestPredLabels, bestDimText, bestParamText] = ...
                                    update_best_result(model, Y, currentDimText, paramText, ...
                                    bestAcc, bestPredLabels, bestDimText, bestParamText);
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
                            [bestAcc, bestPredLabels, bestDimText, bestParamText] = ...
                                update_best_result(model, Y, currentDimText, paramText, ...
                                bestAcc, bestPredLabels, bestDimText, bestParamText);
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
                        [bestAcc, bestPredLabels, bestDimText, bestParamText] = ...
                            update_best_result(model, Y, currentDimText, paramText, ...
                            bestAcc, bestPredLabels, bestDimText, bestParamText);
                    end
                end
            case 'lda'
                for typeIdx = 1:numel(ldaTypeList)
                    model = fitcdiscr(Xmodel, Y, ...
                        'DiscrimType', ldaTypeList{typeIdx}, ...
                        'CVPartition', cvp);
                    paramText = sprintf('type=%s', ldaTypeList{typeIdx});
                    [bestAcc, bestPredLabels, bestDimText, bestParamText] = ...
                        update_best_result(model, Y, currentDimText, paramText, ...
                        bestAcc, bestPredLabels, bestDimText, bestParamText);
                end
        end
end

misIdx = find(bestPredLabels ~= Y);
result.groupIdx = g;
result.groupName = groupNames{g};
result.featureText = strjoin(featNames(featIdx), ', ');
result.accuracy = bestAcc;
result.kfold = kfold;
result.algorithm = algorithm;
result.labelName = labelName;
result.dimText = bestDimText;
result.paramText = bestParamText;
result.misIdx = misIdx;
result.misTrue = Y(misIdx);
result.misPred = bestPredLabels(misIdx);
result.misFiles = allFilePaths(misIdx);

fprintf('[Done] Group %d best accuracy: %.2f%% | %s | %s\n', ...
    g, bestAcc * 100, bestDimText, bestParamText);
drawnow;
end

function [cluster_idx, bestMap, bestDimText, bestScore] = evaluate_unsupervised_mode( ...
    Xn, algorithm, num_clusters, dimConfig)
candidates = build_dimensionality_candidates(Xn, dimConfig, max(2, size(Xn,2)));
if isempty(candidates)
    candidates = build_dimensionality_candidates(Xn, dimConfig, max(2, size(Xn,2)), {'none'});
end

bestScore = -inf;
bestDimText = '';
bestMap = [];
cluster_idx = [];

for idx = 1:numel(candidates)
    Xcluster = candidates(idx).map2d;
    currentDimText = candidates(idx).dimText;
    fprintf('[Unsupervised] Trying dim mode: %s\n', currentDimText);
    drawnow;

    currentCluster = run_unsupervised_clustering(Xcluster, algorithm, num_clusters);
    currentScore = compute_unsupervised_score(Xcluster, currentCluster);

    if currentScore > bestScore
        bestScore = currentScore;
        bestDimText = currentDimText;
        bestMap = Xcluster(:, 1:min(2, size(Xcluster,2)));
        if size(bestMap,2) == 1
            bestMap = [bestMap, zeros(size(bestMap,1),1)];
        end
        cluster_idx = currentCluster;
    end
end
end

function cluster_idx = run_unsupervised_clustering(Xinput, algorithm, num_clusters)
if size(Xinput,2) == 1 && strcmpi(algorithm, 'hierarchical')
    x_sorted = sort(Xinput);
    th = x_sorted(round(length(Xinput)/2));
    cluster_idx = ones(size(Xinput,1),1);
    cluster_idx(Xinput > th) = min(2, num_clusters);
    return;
end

switch lower(algorithm)
    case 'hierarchical'
        D = pdist(Xinput, 'cityblock');
        Z = linkage(D, 'average');
        cluster_idx = cluster(Z, 'maxclust', num_clusters);
    case 'kmeans'
        cluster_idx = kmeans(Xinput, num_clusters, 'Replicates', 50);
    case 'gmm'
        gm = fitgmdist(Xinput, num_clusters, 'Replicates', 20, ...
            'RegularizationValue', 1e-6);
        cluster_idx = cluster(gm, Xinput);
    case 'spectral'
        cluster_idx = spectralcluster(Xinput, num_clusters);
    case 'dbscan'
        cluster_idx = dbscan(Xinput, 0.8, 3);
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

function score = compute_unsupervised_score(Xinput, cluster_idx)
validClusters = unique(cluster_idx);
if numel(validClusters) < 2
    score = -inf;
    return;
end

try
    s = silhouette(Xinput, cluster_idx);
    score = mean(s);
catch
    score = -inf;
end
end

function candidates = build_dimensionality_candidates(Xn, dimConfig, maxLinearDim, overrideModes)
if nargin < 4 || isempty(overrideModes)
    dimModes = dimConfig.modes;
else
    dimModes = overrideModes;
end

pcaDimList = dimConfig.pcaDims;
tsnePerplexityList = dimConfig.tsnePerplexities;
mdsDistance = dimConfig.mdsDistance;
candidates = struct('map2d', {}, 'dimText', {});
nDim = size(Xn, 2);

for modeIdx = 1:numel(dimModes)
    currentMode = lower(dimModes{modeIdx});
    switch currentMode
        case 'none'
            candidates(end+1).map2d = Xn;
            candidates(end).dimText = sprintf('none(%d dims)', nDim);
        case 'pca'
            if nDim < 2
                continue;
            end
            [~, scorePca, ~, ~, explainedPca] = pca(Xn);
            validDims = get_valid_linear_dims(nDim, maxLinearDim, pcaDimList);
            for d = validDims
                candidates(end+1).map2d = scorePca(:,1:d);
                candidates(end).dimText = sprintf('pca(%d dims, %.2f%% var)', ...
                    d, sum(explainedPca(1:d)));
            end
        case 'svd'
            if nDim < 2
                continue;
            end
            [U, S, ~] = svd(Xn, 'econ');
            scoreSvd = U * S;
            validDims = get_valid_linear_dims(size(scoreSvd,2), maxLinearDim, pcaDimList);
            for d = validDims
                candidates(end+1).map2d = scoreSvd(:,1:d);
                candidates(end).dimText = sprintf('svd(%d dims)', d);
            end
        case 'mds'
            if size(Xn,1) < 3
                continue;
            end
            D = pdist(Xn, mdsDistance);
            mapMds = mdscale(D, 2, 'Start', 'cmdscale');
            candidates(end+1).map2d = mapMds;
            candidates(end).dimText = sprintf('mds(2 dims, %s)', mdsDistance);
        case 'tsne'
            if size(Xn,1) < 3
                continue;
            end
            maxPerplexity = max(2, min(size(Xn,1) - 1, floor((size(Xn,1) - 1) / 3)));
            validPerps = unique(tsnePerplexityList(tsnePerplexityList <= maxPerplexity));
            if isempty(validPerps)
                validPerps = maxPerplexity;
            end
            for perp = validPerps
                mapTsne = tsne(Xn, 'NumDimensions', 2, 'Perplexity', perp);
                candidates(end+1).map2d = mapTsne;
                candidates(end).dimText = sprintf('tsne(2 dims, perp=%d)', perp);
            end
        otherwise
            error('Unknown dim reduction mode: %s', currentMode);
    end
end
end

function validDims = get_valid_linear_dims(dataDim, maxLinearDim, pcaDimList)
validDims = unique(pcaDimList(pcaDimList <= min(dataDim, maxLinearDim)));
if isempty(validDims)
    validDims = min(2, min(dataDim, maxLinearDim));
end
end

function [bestAcc, bestPredLabels, bestDimText, bestParamText] = update_best_result( ...
    model, Y, currentDimText, paramText, bestAcc, bestPredLabels, bestDimText, bestParamText)
predLabels = kfoldPredict(model);
acc = mean(predLabels == Y);
if acc > bestAcc
    bestAcc = acc;
    bestPredLabels = predLabels;
    bestDimText = currentDimText;
    bestParamText = paramText;
    fprintf('[Best] acc=%.2f%% | %s | %s\n', acc * 100, bestDimText, bestParamText);
    drawnow;
end
end

function print_top_supervised_results(supervisedResults, topKToPrint)
if isempty(supervisedResults)
    fprintf('\nNo supervised results were generated.\n');
    return;
end

allAcc = [supervisedResults.accuracy];
[~, sortIdx] = sort(allAcc, 'descend');
topN = min(topKToPrint, numel(sortIdx));

fprintf('\n\n===============================================================\n');
fprintf(' Top %d supervised results | Algorithm: %s | Label: %s\n', ...
    topN, supervisedResults(sortIdx(1)).algorithm, supervisedResults(sortIdx(1)).labelName);
fprintf('===============================================================\n');

for i = 1:topN
    result = supervisedResults(sortIdx(i));
    fprintf('\nRank %d | Feature group %d -> %s\n', i, result.groupIdx, result.groupName);
    fprintf('Features: %s\n', result.featureText);
    fprintf('Best setup: %s + %s\n', result.algorithm, result.dimText);
    fprintf('Dim reduction: %s\n', result.dimText);
    fprintf('Best params: %s\n', result.paramText);
    fprintf('%d-fold CV accuracy: %.2f%%\n', result.kfold, result.accuracy * 100);
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

function print_top_unsupervised_results(unsupervisedResults, topKToPrint)
if isempty(unsupervisedResults)
    fprintf('\nNo unsupervised results were generated.\n');
    return;
end

allScores = [unsupervisedResults.silhouette];
[~, sortIdx] = sort(allScores, 'descend');
topN = min(topKToPrint, numel(sortIdx));

fprintf('\n\n===============================================================\n');
fprintf(' Top %d unsupervised results | Algorithm: %s | Label view: %s\n', ...
    topN, unsupervisedResults(sortIdx(1)).algorithm, unsupervisedResults(sortIdx(1)).labelName);
fprintf('===============================================================\n');

for i = 1:topN
    result = unsupervisedResults(sortIdx(i));
    fprintf('\nRank %d | Feature group %d -> %s\n', i, result.groupIdx, result.groupName);
    fprintf('Features: %s\n', result.featureText);
    fprintf('Best setup: %s + %s\n', result.algorithm, result.dimText);
    fprintf('Silhouette: %.4f\n', result.silhouette);
end
end

function print_top_visual_results(visualResults, topKToPrint)
if isempty(visualResults)
    fprintf('\nNo 2D visualization results were generated.\n');
    return;
end

allAcc2d = [visualResults.accuracy];
[~, sortIdx2d] = sort(allAcc2d, 'descend');
topN2d = min(topKToPrint, numel(sortIdx2d));

fprintf('\n\n===============================================================\n');
fprintf(' Top %d 2D visualization results | Label: %s\n', ...
    topN2d, visualResults(sortIdx2d(1)).labelName);
fprintf('===============================================================\n');

for i = 1:topN2d
    result = visualResults(sortIdx2d(i));
    fprintf('\nRank %d | Feature group %d -> %s\n', i, result.groupIdx, result.groupName);
    fprintf('Features: %s\n', result.featureText);
    fprintf('2D projection: %s\n', result.dimText);
    fprintf('2D params: %s\n', result.paramText);
    fprintf('%d-fold 2D KNN accuracy: %.2f%%\n', ...
        min(5, min(arrayfun(@(y)sum(result.labels == y), unique(result.labels)))), ...
        result.accuracy * 100);
    fprintf('Misclassified samples: %d\n', numel(result.misIdx));
    plot_visual_result(result, i);
end
end

function plot_visual_result(result, rankIdx)
fig = figure('Position',[100,100,900,650]);
h = gscatter(result.map2d(:,1), result.map2d(:,2), result.labels);
hold on;
grid on;
title(sprintf('Top %d | Group %d: %s | %.2f%%', ...
    rankIdx, result.groupIdx, result.groupName, result.accuracy * 100), 'FontSize', 14);
xlabel('Dimension 1', 'FontSize', 12);
ylabel('Dimension 2', 'FontSize', 12);

classLabels = unique(result.labels);
for c = 1:numel(classLabels)
    pts = result.map2d(result.labels == classLabels(c), :);
    edgeColor = h(c).Color;
    if size(pts,1) >= 3
        hullIdx = convhull(pts(:,1), pts(:,2));
        patch(pts(hullIdx,1), pts(hullIdx,2), edgeColor, ...
            'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(pts(hullIdx,1), pts(hullIdx,2), '-', 'Color', edgeColor, 'LineWidth', 1.8);
    elseif size(pts,1) == 2
        plot(pts(:,1), pts(:,2), '-', 'Color', edgeColor, 'LineWidth', 1.8);
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
saveas(fig, sprintf('top2d_rank%d_group%d.png', rankIdx, result.groupIdx));
end
%{

fprintf('\n馃帀 鍏ㄩ儴杩愯瀹屾瘯锛乗n');
%}
