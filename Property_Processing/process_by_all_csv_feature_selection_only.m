% 功能：按 all_csv_data.csv 列头先做特征筛选，不立即训练
% 示例：
%   process_by_all_csv_feature_selection_only('a*')
%   process_by_all_csv_feature_selection_only('a*','sg+msc',3,15,'pca',40)
%   process_by_all_csv_feature_selection_only('a*','sg+msc',3,15,'cars',30)
function fs_result = process_by_all_csv_feature_selection_only(property_name, preproc_mode, sg_order, sg_window, fs_method, fs_param)

if nargin < 1 || isempty(property_name)
    error('property_name 必填，例如：''a*''。');
end
if nargin < 2 || isempty(preproc_mode), preproc_mode = 'sg+msc'; end
if nargin < 3 || isempty(sg_order), sg_order = 3; end
if nargin < 4 || isempty(sg_window), sg_window = 15; end
if nargin < 5 || isempty(fs_method), fs_method = 'corr_topk'; end
if nargin < 6, fs_param = []; end

project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'Package', 'Training'), '-begin');
addpath(genpath(fullfile(project_root, 'Package')));

black_dir = fullfile(project_root, 'data', 'Black white');
if exist(fullfile(black_dir, 'black.csv'), 'file')
    black_file_name = 'black.csv';
else
    cands = dir(fullfile(black_dir, '*.csv'));
    if isempty(cands), error('未找到黑白参考文件。'); end
    black_file_name = cands(1).name;
end
csv_folder = fullfile(project_root, 'data', 'NIR');
black_file = fullfile(black_dir, black_file_name);
black_out = fullfile(project_root, 'Result', 'Black_White', 'post_processing_data.xlsx');
if exist(black_out, 'file'), delete(black_out); end
Black_White_Processing(csv_folder, black_file, black_out);

[y, ~] = build_property_vector_from_all_csv('data\physical\all_csv_data.csv', property_name, 'data\NIR', 1);
sample_count = numel(y);

safe_tag = regexprep(char(property_name), '[^a-zA-Z0-9_]', '_');
time_tag = datestr(now, 'yyyymmdd_HHMMSS');
tag_fs = fullfile(safe_tag, ['SELECT_', upper(fs_method), '_', time_tag]);

pre_processing_common(tag_fs, sg_order, sg_window, sample_count, preproc_mode);
smooth_path = fullfile(project_root, 'Result', 'Smooth', tag_fs, 'Smooth_Results.mat');
S = load(smooth_path, 'Post_smooth_data');
X = S.Post_smooth_data(1:sample_count, :);

fs_result = select_features_by_method(X, y, fs_method, fs_param);
X_sel = X(:, fs_result.selected_idx);

fs_dir = fullfile(project_root, 'Result', 'Feature_Select', tag_fs);
if ~exist(fs_dir, 'dir'), mkdir(fs_dir); end

fs_result.property_name = property_name;
fs_result.preproc_mode = preproc_mode;
fs_result.sg_order = sg_order;
fs_result.sg_window = sg_window;
fs_result.sample_count = sample_count;
fs_result.X_selected = X_sel;
fs_result.y = y;

save(fullfile(fs_dir, 'feature_select_result.mat'), 'fs_result');
xlswrite(fullfile(fs_dir, 'selected_idx.xlsx'), fs_result.selected_idx(:));
xlswrite(fullfile(fs_dir, 'selected_data.xlsx'), X_sel);

fig_fs = figure(101);
plot(fs_result.score, 'LineWidth', 1.2); hold on;
scatter(fs_result.selected_idx, fs_result.score(fs_result.selected_idx), 16, 'r', 'filled');
title(sprintf('Feature Selection (%s) - %s', upper(fs_method), property_name));
xlabel('Feature Index'); ylabel('Feature Score');
saveas(fig_fs, fullfile(fs_dir, 'feature_select_score.tif'), 'tiff');
close(fig_fs);

fprintf('特征筛选完成: 属性=%s, 方法=%s, 特征数=%d\n', property_name, fs_method, numel(fs_result.selected_idx));
fprintf('筛选结果目录：Result\\Feature_Select\\%s\\\n', tag_fs);
end


