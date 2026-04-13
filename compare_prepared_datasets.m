function cmp_result = compare_prepared_datasets(before_dataset_mat_path, after_dataset_mat_path)
% 对比两个已保存的数据集，并将图片保存在 after 数据集目录下。

if nargin < 2 || isempty(before_dataset_mat_path) || isempty(after_dataset_mat_path)
    error('Please provide both before and after dataset.mat paths.');
end

S1 = load(before_dataset_mat_path, 'dataset');
S2 = load(after_dataset_mat_path, 'dataset');
if ~isfield(S1, 'dataset') || ~isfield(S2, 'dataset')
    error('dataset variable not found in one of the provided files.');
end

before_ds = S1.dataset;
after_ds = S2.dataset;
X1 = before_ds.X;
X2 = after_ds.X;
y1 = before_ds.y(:);
y2 = after_ds.y(:);

if size(X1, 1) ~= numel(y1) || size(X2, 1) ~= numel(y2)
    error('Sample counts do not match y in one of the datasets.');
end
if size(X1, 1) ~= size(X2, 1)
    error('Before and after datasets must have the same number of samples.');
end

out_dir = fileparts(after_dataset_mat_path);
if isempty(out_dir)
    out_dir = pwd;
end

cmp_result = struct();
cmp_result.before_dataset_mat_path = before_dataset_mat_path;
cmp_result.after_dataset_mat_path = after_dataset_mat_path;
cmp_result.output_dir = out_dir;

before_tag = 'before';
after_tag = 'after';
if isfield(before_ds, 'metadata') && isfield(before_ds.metadata, 'dataset_tag')
    before_tag = strrep(before_ds.metadata.dataset_tag, '\', ' / ');
end
if isfield(after_ds, 'metadata') && isfield(after_ds.metadata, 'dataset_tag')
    after_tag = strrep(after_ds.metadata.dataset_tag, '\', ' / ');
end

% 1) Mean curve comparison
mu1 = mean(X1, 1);
mu2 = mean(X2, 1);
fig1 = figure(501);
set(fig1, 'Name', sprintf('均值曲线对比 - %s -> %s', before_tag, after_tag), 'NumberTitle', 'off');
plot(1:numel(mu1), mu1, 'Color', [0.2 0.2 0.8], 'LineWidth', 1.2); hold on;
plot(1:numel(mu2), mu2, 'Color', [0.9 0.2 0.2], 'LineWidth', 1.2);
hold off;
xlabel('特征索引');
ylabel('均值');
title(sprintf('筛选前后均值曲线对比 | %s -> %s', before_tag, after_tag));
legend('筛选前', '筛选后', 'Location', 'best');
saveas(fig1, fullfile(out_dir, 'cmp_mean_curve.tif'), 'tiff');
cmp_result.cmp_mean_curve = fullfile(out_dir, 'cmp_mean_curve.tif');

% 2) Heatmap comparison
fig2 = figure(502);
set(fig2, 'Name', sprintf('热图对比 - %s -> %s', before_tag, after_tag), 'NumberTitle', 'off');
subplot(1,2,1);
imagesc(X1);
colormap(parula);
colorbar;
title(sprintf('筛选前 | %s', before_tag));
xlabel('特征索引'); ylabel('样本索引');
subplot(1,2,2);
imagesc(X2);
colormap(parula);
colorbar;
title(sprintf('筛选后 | %s', after_tag));
xlabel('特征索引'); ylabel('样本索引');
saveas(fig2, fullfile(out_dir, 'cmp_heatmap.tif'), 'tiff');
cmp_result.cmp_heatmap = fullfile(out_dir, 'cmp_heatmap.tif');

% 3) PCA scatter comparison
fig3 = figure(503);
set(fig3, 'Name', sprintf('主成分散点对比 - %s -> %s', before_tag, after_tag), 'NumberTitle', 'off');
subplot(1,2,1);
if size(X1, 2) >= 2
    [~, score1, ~, ~, explained1] = pca(X1);
    scatter(score1(:, 1), score1(:, 2), 30, y1, 'filled');
    xlabel(sprintf('PC1 (%.2f%%)', explained1(1)));
    ylabel(sprintf('PC2 (%.2f%%)', explained1(2)));
else
    scatter((1:size(X1, 1))', X1(:, 1), 30, y1, 'filled');
    xlabel('样本索引'); ylabel('特征 1');
end
title(sprintf('筛选前主成分散点图 | %s', before_tag));
colorbar;
subplot(1,2,2);
if size(X2, 2) >= 2
    [~, score2, ~, ~, explained2] = pca(X2);
    scatter(score2(:, 1), score2(:, 2), 30, y2, 'filled');
    xlabel(sprintf('PC1 (%.2f%%)', explained2(1)));
    ylabel(sprintf('PC2 (%.2f%%)', explained2(2)));
else
    scatter((1:size(X2, 1))', X2(:, 1), 30, y2, 'filled');
    xlabel('样本索引'); ylabel('特征 1');
end
title(sprintf('筛选后主成分散点图 | %s', after_tag));
colorbar;
saveas(fig3, fullfile(out_dir, 'cmp_pca_scatter.tif'), 'tiff');
cmp_result.cmp_pca_scatter = fullfile(out_dir, 'cmp_pca_scatter.tif');

% 4) Feature count bar
fig4 = figure(504);
set(fig4, 'Name', sprintf('特征数量对比 - %s -> %s', before_tag, after_tag), 'NumberTitle', 'off');
bar([size(X1, 2), size(X2, 2)]);
set(gca, 'XTickLabel', {'筛选前', '筛选后'});
ylabel('特征数量');
title(sprintf('特征数量对比 | 筛选前=%d | 筛选后=%d', size(X1, 2), size(X2, 2)));
saveas(fig4, fullfile(out_dir, 'cmp_feature_count.tif'), 'tiff');
cmp_result.cmp_feature_count = fullfile(out_dir, 'cmp_feature_count.tif');

% 5) Selected feature positions if available in after dataset metadata
if isfield(after_ds, 'metadata') && isfield(after_ds.metadata, 'selected_idx') && ~isempty(after_ds.metadata.selected_idx)
    selected_idx = after_ds.metadata.selected_idx(:)';
    n_before = size(X1, 2);
    y_base = zeros(1, n_before);
    if isfield(after_ds, 'metadata') && isfield(after_ds.metadata, 'feature_score') && ~isempty(after_ds.metadata.feature_score)
        score_all = after_ds.metadata.feature_score(:)';
        n_before = max(n_before, numel(score_all));
        y_base = zeros(1, n_before);
        y_base(1:numel(score_all)) = score_all;
    end
    fig5 = figure(505);
    set(fig5, 'Name', sprintf('筛选特征位置对比 - %s', after_tag), 'NumberTitle', 'off');
    plot(1:numel(y_base), y_base, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.0);
    hold on;
    scatter(selected_idx, y_base(selected_idx), 28, 'r', 'filled');
    hold off;
    xlabel('原始特征索引');
    ylabel('得分');
    title(sprintf('筛选后特征在原始特征空间中的位置 | %s | 已选特征数=%d', after_tag, numel(selected_idx)));
    saveas(fig5, fullfile(out_dir, 'cmp_selected_positions.tif'), 'tiff');
    cmp_result.cmp_selected_positions = fullfile(out_dir, 'cmp_selected_positions.tif');
end

save(fullfile(out_dir, 'compare_result.mat'), 'cmp_result');
fprintf('Comparison figures saved to: %s\n', out_dir);
end
