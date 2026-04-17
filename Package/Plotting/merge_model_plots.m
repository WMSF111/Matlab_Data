function output = merge_model_plots(run_name_or_dir, model_name)
% 功能：将某次运行中同一回归器下的所有回归图拼成一张大图，便于对比
% 用法：
%   output = merge_model_plots('Run_20260414_122245', 'GPR')
%   output = merge_model_plots('D:\HXR\Matlab\Result\Model\Run_20260414_122245', 'GPR')

if nargin < 1 || isempty(run_name_or_dir)
    error('请提供运行目录名称或完整路径，例如：Run_20260414_122245。');
end
if nargin < 2 || isempty(model_name)
    error('请提供回归器名称，例如：GPR。');
end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if contains(run_name_or_dir, ':') || startsWith(run_name_or_dir, filesep)
    run_dir = run_name_or_dir;
else
    run_dir = fullfile(project_root, 'Result', 'Model', run_name_or_dir);
end

model_dir = fullfile(run_dir, upper(model_name));
if ~exist(model_dir, 'dir')
    error('未找到模型目录：%s', model_dir);
end

files = dir(fullfile(model_dir, '**', '*.tif'));
if isempty(files)
    error('目录中未找到回归图像：%s', model_dir);
end

n = numel(files);
cols = ceil(sqrt(n));
rows = ceil(n / cols);

fig = figure('Name', ['回归图总览 - ' upper(model_name)], 'NumberTitle', 'off', 'Color', 'w');
set(fig, 'Position', [100, 60, 1600, 900]);

for i = 1:n
    ax = subplot(rows, cols, i);
    img_path = fullfile(files(i).folder, files(i).name);
    img = imread(img_path);
    image(ax, img);
    axis(ax, 'image');
    axis(ax, 'off');

    [~, folder_name] = fileparts(files(i).folder);
    title(ax, strrep(folder_name, '_', '\_'), 'Interpreter', 'none', 'FontSize', 9);
end

sgtitle(sprintf('运行 %s | 回归器 %s 图像对比', strrep(get_last_path_part(run_dir), '_', '\_'), upper(model_name)), 'Interpreter', 'none');

output_dir = fullfile(run_dir, upper(model_name));
output.image_path = fullfile(output_dir, ['combined_' upper(model_name) '_plots.tif']);
output.fig_path = fullfile(output_dir, ['combined_' upper(model_name) '_plots.fig']);
output.file_count = n;
output.model_dir = model_dir;

saveas(fig, output.image_path, 'tiff');
savefig(fig, output.fig_path);

fprintf('已生成回归图总览：%s\n', output.image_path);
fprintf('图像数量：%d\n', n);
end

function name = get_last_path_part(path_str)
parts = regexp(path_str, '[\\/]+', 'split');
parts = parts(~cellfun(@isempty, parts));
name = parts{end};
end
