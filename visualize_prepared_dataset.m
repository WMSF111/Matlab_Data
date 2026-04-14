function vis_result = visualize_prepared_dataset(dataset_mat_path)
% 可视化已保存的数据集，并将图片保存在 dataset.mat 同目录下。

if nargin < 1 || isempty(dataset_mat_path)
    error('Please provide dataset.mat path.');
end

S = load(dataset_mat_path, 'dataset');
if ~isfield(S, 'dataset')
    error('dataset variable not found in: %s', dataset_mat_path);
end

dataset = S.dataset;
X = dataset.X;
y = dataset.y(:);

if size(X, 1) ~= numel(y)
    error('X and y sample counts do not match.');
end

out_dir = fileparts(dataset_mat_path);
if isempty(out_dir)
    out_dir = pwd;
end

vis_result = struct();
vis_result.dataset_mat_path = dataset_mat_path;
vis_result.output_dir = out_dir;

dataset_tag = 'dataset';
stage_tag = '';
if isfield(dataset, 'metadata')
    if isfield(dataset.metadata, 'dataset_tag') && ~isempty(dataset.metadata.dataset_tag)
        dataset_tag = strrep(dataset.metadata.dataset_tag, '\', ' / ');
    end
    if isfield(dataset.metadata, 'data_stage') && ~isempty(dataset.metadata.data_stage)
        stage_tag = char(dataset.metadata.data_stage);
    end
end

% 1) Heatmap of samples x features
fig1 = figure(401);
set(fig1, 'Name', sprintf('数据集热图 - %s', dataset_tag), 'NumberTitle', 'off');
imagesc(X);
colormap(parula);
colorbar;
xlabel('特征索引');
ylabel('样本索引');
title(sprintf('数据集热图 | %s | 阶段=%s | 样本=%d | 特征=%d', ...
    dataset_tag, stage_tag, size(X, 1), size(X, 2)));
saveas(fig1, fullfile(out_dir, 'vis_heatmap.tif'), 'tiff');
vis_result.vis_heatmap = fullfile(out_dir, 'vis_heatmap.tif');

% 2) Mean and std envelope across features
mu = mean(X, 1);
sigma = std(X, 0, 1);
xx = 1:size(X, 2);
fig2 = figure(402);
set(fig2, 'Name', sprintf('均值与标准差 - %s', dataset_tag), 'NumberTitle', 'off');
fill([xx, fliplr(xx)], [mu - sigma, fliplr(mu + sigma)], [0.85 0.92 1.00], ...
    'EdgeColor', 'none');
hold on;
plot(xx, mu, 'b-', 'LineWidth', 1.5);
hold off;
xlabel('特征索引');
ylabel('数值');
title(sprintf('均值曲线与标准差带 | %s | 阶段=%s', dataset_tag, stage_tag));
saveas(fig2, fullfile(out_dir, 'vis_mean_std.tif'), 'tiff');
vis_result.vis_mean_std = fullfile(out_dir, 'vis_mean_std.tif');

% 3) PCA scatter of samples colored by y
fig3 = figure(403);
set(fig3, 'Name', sprintf('主成分散点图 - %s', dataset_tag), 'NumberTitle', 'off');
if size(X, 2) >= 2
    [~, score, ~, ~, explained] = pca(X);
    scatter(score(:, 1), score(:, 2), 36, y, 'filled');
    xlabel(sprintf('PC1 (%.2f%%)', explained(1)));
    ylabel(sprintf('PC2 (%.2f%%)', explained(2)));
    title(sprintf('主成分散点图（按 y 着色） | %s | 阶段=%s', dataset_tag, stage_tag));
    colorbar;
else
    scatter((1:size(X, 1))', X(:, 1), 36, y, 'filled');
    xlabel('样本索引');
    ylabel('特征 1');
    title(sprintf('单特征散点图（按 y 着色） | %s | 阶段=%s', dataset_tag, stage_tag));
    colorbar;
end
saveas(fig3, fullfile(out_dir, 'vis_pca_scatter.tif'), 'tiff');
vis_result.vis_pca_scatter = fullfile(out_dir, 'vis_pca_scatter.tif');

% 4) Selected feature positions if available
if isfield(dataset, 'metadata') && isfield(dataset.metadata, 'selected_idx') && ~isempty(dataset.metadata.selected_idx)
    selected_idx = dataset.metadata.selected_idx(:)';
    feature_count = selected_idx(end);
    if isfield(dataset.metadata, 'feature_score') && ~isempty(dataset.metadata.feature_score)
        score = dataset.metadata.feature_score(:)';
        feature_count = max(feature_count, numel(score));
    else
        feature_count = max(feature_count, max(selected_idx));
        score = zeros(1, feature_count);
        score(selected_idx) = 1;
    end

    fig4 = figure(404);
    set(fig4, 'Name', sprintf('筛选特征位置 - %s', dataset_tag), 'NumberTitle', 'off');
    plot(1:numel(score), score, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.0);
    hold on;
    scatter(selected_idx, score(selected_idx), 26, 'r', 'filled');
    hold off;
    xlabel('原始特征索引');
    ylabel('得分');
    title(sprintf('筛选特征位置 | %s | 已选特征数=%d', dataset_tag, numel(selected_idx)));
    saveas(fig4, fullfile(out_dir, 'vis_selected_idx.tif'), 'tiff');
    vis_result.vis_selected_idx = fullfile(out_dir, 'vis_selected_idx.tif');
end

save(fullfile(out_dir, 'visualization_result.mat'), 'vis_result');
fprintf('Visualization saved to: %s\n', out_dir);
end
