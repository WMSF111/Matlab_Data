function dataset = prepare_property_dataset(property_name, data_stage, preproc_mode, sg_order, sg_window, fs_method, fs_param, msc_ref_mode, snv_mode, keep_exports, baseline_zero_mode, despike_mode, baseline_zero_scope)
% Prepare dataset for modeling.
% This version only prepares black-corrected raw spectra and metadata.
% Actual preprocessing / feature selection is performed after train-test split.

if nargin < 1 || isempty(property_name)
    error('property_name is required, for example ''a*''.' );
end
if nargin < 2 || isempty(data_stage), data_stage = 'raw'; end
if nargin < 3 || isempty(preproc_mode), preproc_mode = 'sg+msc'; end
if nargin < 4 || isempty(sg_order), sg_order = 3; end
if nargin < 5 || isempty(sg_window), sg_window = 15; end
if nargin < 6 || isempty(fs_method), fs_method = 'corr_topk'; end
if nargin < 7, fs_param = []; end
if nargin < 8 || isempty(msc_ref_mode), msc_ref_mode = 'mean'; end
if nargin < 9 || isempty(snv_mode), snv_mode = 'standard'; end
if nargin < 10 || isempty(keep_exports), keep_exports = false; end
if nargin < 11 || isempty(baseline_zero_mode), baseline_zero_mode = 'none'; end
if nargin < 12 || isempty(despike_mode), despike_mode = 'none'; end
if nargin < 13 || isempty(baseline_zero_scope), baseline_zero_scope = 'cropped_spectrum'; end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(project_root));

black_dir = fullfile(project_root, 'data', 'Black white');
if exist(fullfile(black_dir, 'black.csv'), 'file')
    black_file_name = 'black.csv';
else
    cands = dir(fullfile(black_dir, '*.csv'));
    if isempty(cands)
        error('Black/white reference file was not found.');
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
X_full = readmatrix(black_out);
X_full = X_full(1:sample_count, :);
source_feature_count = size(X_full, 2);

cut_left = 0;
cut_right = 0;
used_band_idx_full = (1 + cut_left):(source_feature_count - cut_right);

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
metadata.baseline_zero_mode = baseline_zero_mode;
metadata.despike_mode = despike_mode;
metadata.baseline_zero_scope = baseline_zero_scope;
metadata.keep_exports = keep_exports;
metadata.cut_left = cut_left;
metadata.cut_right = cut_right;
metadata.source_feature_count = source_feature_count;
metadata.used_band_idx = used_band_idx_full;
metadata.used_band_range = [used_band_idx_full(1), used_band_idx_full(end)];
metadata.preprocess_after_split = true;

switch stage
    case 'raw'
        dataset_tag = fullfile(safe_tag, ['RAW_', time_tag]);
    case 'preprocessed'
        dataset_tag = fullfile(safe_tag, ['PREPROCESSED_', time_tag]);
    case 'selected'
        dataset_tag = fullfile(safe_tag, ['SELECTED_', upper(fs_method), '_', time_tag]);
        metadata.selection_applied_in_training = true;
    otherwise
        error('Unsupported data_stage: %s. Use raw / preprocessed / selected.', data_stage);
end

dataset = save_prepared_dataset(dataset_tag, X_full, y, metadata, keep_exports);

label_prepared = char([25968,25454,38598,24050,20934,22791]);
label_stage = char([25968,25454,38454,27573]);
label_samples = char([26679,26412,25968]);
label_features = char([29305,24449,25968]);
label_band_range = char([39044,35745,20351,29992,27874,27573,33539,22260]);
label_band_count = char([39044,35745,20351,29992,27874,27573,25968]);
label_msc_mode = char([77,83,67,32,27169,24335]);
label_snv_mode = char([83,78,86,32,27169,24335]);
label_baseline = char([22522,32447,24402,38646]);
label_despike = char([21435,23574,21050,27169,24335]);
label_export_mode = char([25968,25454,38598,23548,20986,27169,24335]);
label_split_first = char([20808,21010,35757,27979,21518,35757,32451,38598,24050]);
text_keep = char([20445,30041,32,88,32,47,32,89,32,47,32,100,97,116,97,115,101,116,46,109,97,116]);
text_temp = char([20020,26102,32,100,97,116,97,115,101,116,46,109,97,116,65292,36816,34892,21518,21487,28165,29702]);
text_yes = char([26159]);

fprintf('%s: %s\n', label_prepared, dataset.paths.mat);
fprintf('%s=%s, %s=%d, %s=%d\n', label_stage, stage, label_samples, size(dataset.X, 1), label_features, size(dataset.X, 2));
fprintf('%s=%d:%d | %s=%d\n', label_band_range, metadata.used_band_range(1), metadata.used_band_range(2), label_band_count, numel(metadata.used_band_idx));
fprintf('%s=%s, %s=%s, %s=%s, %s=%s\n', label_msc_mode, msc_ref_mode, label_snv_mode, snv_mode, label_baseline, [baseline_zero_mode ' / ' baseline_zero_scope], label_despike, despike_mode);
fprintf('%s: %s\n', label_split_first, text_yes);
if keep_exports
    fprintf('%s: %s\n', label_export_mode, text_keep);
else
    fprintf('%s: %s\n', label_export_mode, text_temp);
end
end
