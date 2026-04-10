% 功能：光谱预处理公共函数
% 输入：
%   smooth_tag   - 结果标签（如 N、PH、SSC 或自定义）
%   sg_order     - SG 多项式阶数
%   sg_window    - SG 窗口长度
%   sample_count - 参与建模的样本数
%   preproc_mode - 预处理模式：'sg+msc'(默认)、'sg+snv'、'sg+msc+snv'、'sg'、'none'
function pre_processing_common(smooth_tag, sg_order, sg_window, sample_count, preproc_mode)

if nargin < 1 || isempty(smooth_tag), smooth_tag = 'N'; end
if nargin < 2 || isempty(sg_order), sg_order = 3; end
if nargin < 3 || isempty(sg_window), sg_window = 15; end
if nargin < 4 || isempty(sample_count), sample_count = 120; end
if nargin < 5 || isempty(preproc_mode), preproc_mode = 'sg+msc'; end

project_root = fileparts(fileparts(mfilename('fullpath')));
wave_folder = fullfile(project_root, 'data', 'wavelength144.mat');
black_white_data_folder = fullfile(project_root, 'Result', 'Black_White', 'post_processing_data.xlsx');
smooth_mat_file_name = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.mat');
smooth_file_name = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.tif');
smooth_fig_file_name = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.fig');

smooth_dir = fileparts(smooth_mat_file_name);
if ~exist(smooth_dir, 'dir')
    mkdir(smooth_dir);
end

data = xlsread(black_white_data_folder);
data = data(1:sample_count, :);

% 若波长长度不匹配，则使用特征索引作为横轴
x_axis = 1:size(data, 2);
if exist(wave_folder, 'file')
    S = load(wave_folder);
    if isfield(S, 'wavelength144')
        w = S.wavelength144(:)';
        if numel(w) == size(data, 2)
            x_axis = w;
        end
    end
end

% 三种单独滤波结果（用于单独展示）
SG_only = SG(data, sg_order, sg_window);
MSC_only = MSC(data);
SNV_only = SNV(data);

% 按模式处理（最终训练数据仅使用 Post_smooth_data）
POST_data = data;
post_title = 'none';
use_sg = false; use_msc = false; use_snv = false;

switch lower(strtrim(preproc_mode))
    case 'none'
        POST_data = data;
        post_title = 'none';
    case 'sg'
        use_sg = true;
        POST_data = SG_only;
        post_title = 'SG';
    case 'sg+snv'
        use_sg = true; use_snv = true;
        POST_data = SNV(SG_only);
        post_title = 'SG+SNV';
    case 'sg+msc+snv'
        use_sg = true; use_msc = true; use_snv = true;
        POST_data = SNV(MSC(SG_only));
        post_title = 'SG+MSC+SNV';
    otherwise % 'sg+msc'
        use_sg = true; use_msc = true;
        POST_data = MSC(SG_only);
        post_title = 'SG+MSC';
end

Post_smooth_data = POST_data;

% 绘图：5张图（原始 + 3种单独滤波 + 最终结果）
fig_smooth = figure(1);
subplot(2,3,1); plot(x_axis, data'); title('source data');

subplot(2,3,2);
if use_sg
    plot(x_axis, SG_only'); title('SG only');
else
    axis off; title('SG only (unused)');
end

subplot(2,3,3);
if use_msc
    plot(x_axis, MSC_only'); title('MSC only');
else
    axis off; title('MSC only (unused)');
end

subplot(2,3,4);
if use_snv
    plot(x_axis, SNV_only'); title('SNV only');
else
    axis off; title('SNV only (unused)');
end

subplot(2,3,5); plot(x_axis, POST_data'); title(['final used: ', post_title]);
subplot(2,3,6); axis off;

save(smooth_mat_file_name, 'Post_smooth_data', 'preproc_mode', 'sg_order', 'sg_window');
saveas(fig_smooth, smooth_file_name, 'tiff');
savefig(fig_smooth, smooth_fig_file_name);
close(fig_smooth);

fprintf('预处理模式: %s\n', preproc_mode);
fprintf('预处理结果已保存:\n%s\n%s\n%s\n', smooth_mat_file_name, smooth_file_name, smooth_fig_file_name);
end
