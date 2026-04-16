function pre_processing_common(smooth_tag, sg_order, sg_window, sample_count, preproc_mode, msc_ref_mode, snv_mode, cut_left, cut_right, baseline_zero_mode, despike_mode, baseline_zero_scope)
% 功能：光谱预处理公共函数
% 输入：
%   smooth_tag          - 结果标签
%   sg_order            - SG 多项式阶数
%   sg_window           - SG 窗口长度
%   sample_count        - 参与建模的样本数
%   preproc_mode        - 预处理模式：''sg+msc'' / ''sg+snv'' / ''sg+msc+snv'' / ''sg'' / ''none''
%   msc_ref_mode        - MSC 参考模式：''mean'' / ''median'' / ''first''
%   snv_mode            - SNV 模式：''standard'' / ''robust''
%   cut_left            - 左侧裁剪波段数
%   cut_right           - 右侧裁剪波段数
%   baseline_zero_mode  - 基线归零方式：''none'' / ''first_point'' / ''first_5_mean''
%   despike_mode        - 去尖刺方式：''none'' / ''median3'' / ''median5'' / ''median7'' / ''local'' / ''local_strong'' / ''jump_guard''
%   baseline_zero_scope - 基线归零范围：''cropped_spectrum'' / ''full_spectrum''

if nargin < 1 || isempty(smooth_tag), smooth_tag = 'N'; end
if nargin < 2 || isempty(sg_order), sg_order = 3; end
if nargin < 3 || isempty(sg_window), sg_window = 15; end
if nargin < 4 || isempty(sample_count), sample_count = 120; end
if nargin < 5 || isempty(preproc_mode), preproc_mode = 'sg+msc'; end
if nargin < 6 || isempty(msc_ref_mode), msc_ref_mode = 'mean'; end
if nargin < 7 || isempty(snv_mode), snv_mode = 'standard'; end
if nargin < 8 || isempty(cut_left), cut_left = 0; end
if nargin < 9 || isempty(cut_right), cut_right = 0; end
if nargin < 10 || isempty(baseline_zero_mode), baseline_zero_mode = 'none'; end
if nargin < 11 || isempty(despike_mode), despike_mode = 'none'; end
if nargin < 12 || isempty(baseline_zero_scope), baseline_zero_scope = 'cropped_spectrum'; end

project_root = fileparts(fileparts(mfilename('fullpath')));
wave_folder = fullfile(project_root, 'data', 'wavelength144.mat');
black_white_data_folder = fullfile(project_root, 'Result', 'Black_White', 'post_processing_data.csv');
smooth_mat_file_name = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.mat');
smooth_file_name = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.tif');
smooth_fig_file_name = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.fig');

smooth_dir = fileparts(smooth_mat_file_name);
if ~exist(smooth_dir, 'dir')
    mkdir(smooth_dir);
end

full_data = readmatrix(black_white_data_folder);
full_data = full_data(1:sample_count, :);
source_feature_count = size(full_data, 2);
used_band_idx = (1 + cut_left):(source_feature_count - cut_right);

full_data = local_apply_despike(full_data, despike_mode);
if strcmpi(strtrim(baseline_zero_scope), 'full_spectrum')
    full_data = local_apply_baseline_zero(full_data, baseline_zero_mode);
    apply_cropped_baseline = false;
else
    apply_cropped_baseline = true;
end

data = full_data(:, used_band_idx);

x_axis = local_load_x_axis(project_root, wave_folder, source_feature_count);
x_axis = x_axis(used_band_idx);

raw_zero = local_apply_baseline_if_needed(data, baseline_zero_mode, apply_cropped_baseline);
SG_only = local_apply_baseline_if_needed(SG(data, sg_order, sg_window), baseline_zero_mode, apply_cropped_baseline);
MSC_only = local_apply_baseline_if_needed(MSC(data, msc_ref_mode), baseline_zero_mode, apply_cropped_baseline);
SNV_only = local_apply_baseline_if_needed(SNV(data, snv_mode), baseline_zero_mode, apply_cropped_baseline);

POST_data = raw_zero;
post_title = 'none';
use_sg = false; use_msc = false; use_snv = false;

switch lower(strtrim(preproc_mode))
    case 'none'
        POST_data = raw_zero;
        post_title = 'none';
    case 'sg'
        use_sg = true;
        POST_data = SG_only;
        post_title = 'SG';
    case 'sg+snv'
        use_sg = true; use_snv = true;
        POST_data = local_apply_baseline_if_needed(SNV(SG(data, sg_order, sg_window), snv_mode), baseline_zero_mode, apply_cropped_baseline);
        post_title = ['SG+SNV(', snv_mode, ')'];
    case 'sg+msc+snv'
        use_sg = true; use_msc = true; use_snv = true;
        POST_data = local_apply_baseline_if_needed(SNV(MSC(SG(data, sg_order, sg_window), msc_ref_mode), snv_mode), baseline_zero_mode, apply_cropped_baseline);
        post_title = ['SG+MSC(', msc_ref_mode, ')+SNV(', snv_mode, ')'];
    otherwise
        use_sg = true; use_msc = true;
        POST_data = local_apply_baseline_if_needed(MSC(SG(data, sg_order, sg_window), msc_ref_mode), baseline_zero_mode, apply_cropped_baseline);
        post_title = ['SG+MSC(', msc_ref_mode, ')'];
end

Post_smooth_data = POST_data;

fig_smooth = figure(1);
subplot(2,3,1); plot(x_axis, raw_zero'); title('source data');

subplot(2,3,2);
if use_sg
    plot(x_axis, SG_only'); title('SG only');
else
    axis off; title('SG only (unused)');
end

subplot(2,3,3);
if use_msc
    plot(x_axis, MSC_only'); title(['MSC only (', msc_ref_mode, ')']);
else
    axis off; title('MSC only (unused)');
end

subplot(2,3,4);
if use_snv
    plot(x_axis, SNV_only'); title(['SNV only (', snv_mode, ')']);
else
    axis off; title('SNV only (unused)');
end

subplot(2,3,5); plot(x_axis, POST_data'); title(['final used: ', post_title]);
subplot(2,3,6); axis off;

save(smooth_mat_file_name, 'Post_smooth_data', 'preproc_mode', 'sg_order', 'sg_window', 'msc_ref_mode', 'snv_mode', 'cut_left', 'cut_right', 'used_band_idx', 'x_axis', 'baseline_zero_mode', 'despike_mode', 'baseline_zero_scope');
saveas(fig_smooth, smooth_file_name, 'tiff');
savefig(fig_smooth, smooth_fig_file_name);
close(fig_smooth);

fprintf('预处理模式: %s\n', preproc_mode);
fprintf('波段裁剪: 左=%d | 右=%d | 使用范围=%d:%d\n', cut_left, cut_right, used_band_idx(1), used_band_idx(end));
fprintf('MSC 模式: %s | SNV 模式: %s | 基线归零: %s | 归零范围: %s | 去尖刺: %s\n', msc_ref_mode, snv_mode, baseline_zero_mode, baseline_zero_scope, despike_mode);
fprintf('预处理结果已保存:\n%s\n%s\n%s\n', smooth_mat_file_name, smooth_file_name, smooth_fig_file_name);
end

function X_out = local_apply_baseline_if_needed(X_in, mode, apply_flag)
if nargin < 3 || ~apply_flag
    X_out = X_in;
else
    X_out = local_apply_baseline_zero(X_in, mode);
end
end

function X_out = local_apply_baseline_zero(X_in, mode)
X_out = X_in;
if isempty(X_in)
    return;
end
switch lower(strtrim(mode))
    case 'none'
        return;
    case 'first_point'
        base = X_in(:, 1);
    case 'first_5_mean'
        n = min(5, size(X_in, 2));
        base = mean(X_in(:, 1:n), 2, 'omitnan');
    otherwise
        error('不支持的 baseline_zero_mode：%s。可选 none / first_point / first_5_mean。', mode);
end
X_out = X_in - base;
end

function X_out = local_apply_despike(X_in, mode)
X_out = X_in;
if isempty(X_in)
    return;
end
switch lower(strtrim(mode))
    case 'none'
        return;
    case 'median3'
        X_out = medfilt1(X_in, 3, [], 2, 'truncate');
    case 'median5'
        X_out = medfilt1(X_in, 5, [], 2, 'truncate');
    case 'median7'
        X_out = medfilt1(X_in, 7, [], 2, 'truncate');
    case 'local'
        X_out = local_despike_isolated(X_in, 3);
    case 'local_strong'
        X_out = local_despike_isolated(X_in, 2);
    case 'jump_guard'
        X_out = local_jump_guard(X_in);
    otherwise
        error('不支持的 despike_mode：%s。可选 none / median3 / median5 / median7 / local / local_strong / jump_guard。', mode);
end
end

function X_out = local_despike_isolated(X_in, thresh_scale)
X_out = X_in;
if nargin < 2 || isempty(thresh_scale), thresh_scale = 3; end
if size(X_in, 2) < 3
    return;
end
for r = 1:size(X_in, 1)
    row = X_in(r, :);
    for c = 2:(numel(row) - 1)
        left_v = row(c - 1);
        mid_v = row(c);
        right_v = row(c + 1);
        neigh_mean = (left_v + right_v) / 2;
        neigh_diff = abs(right_v - left_v);
        spike_mag = abs(mid_v - neigh_mean);
        local_scale = max([neigh_diff, abs(mid_v - left_v), abs(mid_v - right_v), eps]);
        if spike_mag > thresh_scale * local_scale && sign(mid_v - left_v) ~= sign(right_v - mid_v)
            row(c) = neigh_mean;
        end
    end
    X_out(r, :) = row;
end
end

function X_out = local_jump_guard(X_in)
X_out = X_in;
if size(X_in, 2) < 3
    return;
end
for r = 1:size(X_in, 1)
    row = X_in(r, :);
    for c = 2:(numel(row) - 1)
        left_v = row(c - 1);
        mid_v = row(c);
        right_v = row(c + 1);
        d1 = abs(mid_v - left_v);
        d2 = abs(mid_v - right_v);
        neigh_gap = abs(left_v - right_v);
        jump_ref = max([neigh_gap, eps]);
        if d1 > 4 * jump_ref && d2 > 4 * jump_ref
            row(c) = (left_v + right_v) / 2;
        end
    end
    X_out(r, :) = row;
end
end

function x_axis = local_load_x_axis(project_root, wave_folder, expected_len)
x_axis = 1:expected_len;
nir_dir = fullfile(project_root, 'data', 'NIR');
csv_files = dir(fullfile(nir_dir, '*.csv'));
if ~isempty(csv_files)
    raw = readmatrix(fullfile(nir_dir, csv_files(1).name));
    if ~isempty(raw) && size(raw, 2) >= 2
        axis_col = raw(:, 1);
        axis_col = axis_col(isfinite(axis_col));
        if numel(axis_col) == expected_len
            x_axis = axis_col(:)';
            return;
        end
    end
end
if exist(wave_folder, 'file')
    S = load(wave_folder);
    if isfield(S, 'wavelength144')
        w = S.wavelength144(:)';
        if numel(w) == expected_len
            x_axis = w;
        end
    end
end
end