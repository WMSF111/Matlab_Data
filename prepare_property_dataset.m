function dataset = prepare_property_dataset(property_name, data_stage, preproc_mode, sg_order, sg_window, fs_method, fs_param, msc_ref_mode, snv_mode)
% 功能：准备并保存数据集，不做训练
% data_stage:
%   'raw'          - 黑白校正后的原始数据
%   'preprocessed' - 预处理后的数据
%   'selected'     - 预处理后再做特征筛选的数据
%
% fs_method 说明：
%   'pca'       - 无监督训练前降维/筛选
%   'corr_topk' - 监督式训练前预筛选
%   'spa'       - 偏训练前筛选，减少冗余
%   'cars'      - 建模驱动筛选，和训练强相关
%
% msc_ref_mode 说明：
%   'mean' / 'median' / 'first'
%
% snv_mode 说明：
%   'standard' / 'robust'

if nargin < 1 || isempty(property_name)
    error('property_name 必填，例如：''a*''。');
end
if nargin < 2 || isempty(data_stage), data_stage = 'raw'; end
if nargin < 3 || isempty(preproc_mode), preproc_mode = 'sg+msc'; end
if nargin < 4 || isempty(sg_order), sg_order = 3; end
if nargin < 5 || isempty(sg_window), sg_window = 15; end
if nargin < 6 || isempty(fs_method), fs_method = 'corr_topk'; end
if nargin < 7, fs_param = []; end
if nargin < 8 || isempty(msc_ref_mode), msc_ref_mode = 'mean'; end
if nargin < 9 || isempty(snv_mode), snv_mode = 'standard'; end

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

black_dir = fullfile(project_root, 'data', 'Black white');
if exist(fullfile(black_dir, 'black.csv'), 'file')
    black_file_name = 'black.csv';
else
    cands = dir(fullfile(black_dir, '*.csv'));
    if isempty(cands)
        error('未找到黑白参考文件。');
    end
    black_file_name = cands(1).name;
end
csv_folder = fullfile(project_root, 'data', 'NIR');
black_file = fullfile(black_dir, black_file_name);
black_out = fullfile(project_root, 'Result', 'Black_White', 'post_processing_data.csv');
if exist(black_out, 'file'), delete(black_out); end
Black_White_Processing(csv_folder, black_file, black_out);

[y, ~] = build_property_vector_from_all_csv('data\physical\all_csv_data.csv', property_name, 'data\NIR', 1);
sample_count = numel(y);
X_raw = readmatrix(black_out);
X_raw = X_raw(1:sample_count, :);

safe_tag = regexprep(char(property_name), '[^a-zA-Z0-9_]', '_');
time_tag = datestr(now, 'yyyymmdd_HHMMSS');
stage = lower(strtrim(data_stage));

metadata = struct();
metadata.property_name = property_name;
metadata.data_stage = stage;
metadata.preproc_mode = preproc_mode;
metadata.sg_order = sg_order;
metadata.sg_window = sg_window;
metadata.fs_method = fs_method;
metadata.fs_param = fs_param;
metadata.msc_ref_mode = msc_ref_mode;
metadata.snv_mode = snv_mode;

switch stage
    case 'raw'
        X = X_raw;
        dataset_tag = fullfile(safe_tag, ['RAW_', time_tag]);

    case 'preprocessed'
        prep_tag = fullfile(safe_tag, ['PRE_', time_tag]);
        pre_processing_common(prep_tag, sg_order, sg_window, sample_count, preproc_mode, msc_ref_mode, snv_mode);
        S = load(fullfile(project_root, 'Result', 'Smooth', prep_tag, 'Smooth_Results.mat'), 'Post_smooth_data');
        X = S.Post_smooth_data(1:sample_count, :);
        dataset_tag = fullfile(safe_tag, ['PREPROCESSED_', time_tag]);
        metadata.preprocess_tag = prep_tag;

    case 'selected'
        prep_tag = fullfile(safe_tag, ['SELECT_PRE_', time_tag]);
        pre_processing_common(prep_tag, sg_order, sg_window, sample_count, preproc_mode, msc_ref_mode, snv_mode);
        S = load(fullfile(project_root, 'Result', 'Smooth', prep_tag, 'Smooth_Results.mat'), 'Post_smooth_data');
        X_pre = S.Post_smooth_data(1:sample_count, :);
        fs_result = select_features_by_method(X_pre, y, fs_method, fs_param);
        X = X_pre(:, fs_result.selected_idx);
        dataset_tag = fullfile(safe_tag, ['SELECTED_', upper(fs_method), '_', time_tag]);
        metadata.preprocess_tag = prep_tag;
        metadata.selected_idx = fs_result.selected_idx;
        metadata.feature_score = fs_result.score;
        metadata.feature_info = fs_result.info;

    otherwise
        error('Unsupported data_stage: %s. Use raw / preprocessed / selected.', data_stage);
end

dataset = save_prepared_dataset(dataset_tag, X, y, metadata);

if isfield(metadata, 'feature_score') && isfield(metadata, 'selected_idx')
    fig_fs = figure(201);
    plot(metadata.feature_score, 'LineWidth', 1.2); hold on;
    scatter(metadata.selected_idx, metadata.feature_score(metadata.selected_idx), 16, 'r', 'filled');
    title(sprintf('Feature Selection (%s) - %s', upper(fs_method), property_name));
    xlabel('Feature Index'); ylabel('Feature Score');
    saveas(fig_fs, fullfile(dataset.paths.dir, 'feature_select_score.tif'), 'tiff');
    close(fig_fs);
end

fprintf('数据集已保存: %s\n', dataset.paths.mat);
fprintf('数据阶段=%s, 样本数=%d, 特征数=%d\n', stage, size(dataset.X, 1), size(dataset.X, 2));
fprintf('MSC 模式=%s, SNV 模式=%s\n', msc_ref_mode, snv_mode);
end
