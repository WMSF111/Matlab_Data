function result = show_all_csv_rgb_grid(csv_path, save_path)
% 将 all_csv_data.csv 中所有 L*a*b* 样本转换成 RGB，并显示在一张交互图里
% 鼠标移动到颜色块上时，会显示对应 csv_name、L*a*b* 和 RGB

if nargin < 1 || isempty(csv_path)
    csv_path = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'data', 'physical', 'all_csv_data.csv');
end
if nargin < 2 || isempty(save_path)
    save_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'Result', 'Summary');
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    save_path = fullfile(save_dir, 'all_csv_rgb_grid.tif');
end

T = readtable(csv_path, 'VariableNamingRule', 'preserve');
required_names = {'L*', 'a*', 'b*', 'csv_name'};
for i = 1:numel(required_names)
    if ~ismember(required_names{i}, T.Properties.VariableNames)
        error('CSV 中缺少列: %s', required_names{i});
    end
end

L = double(T{:, 'L*'});
a = double(T{:, 'a*'});
b = double(T{:, 'b*'});
csv_names = string(T{:, 'csv_name'});
valid = isfinite(L) & isfinite(a) & isfinite(b);

lab = [L(valid), a(valid), b(valid)];
rgb = lab2rgb(lab, 'OutputType', 'double');
rgb = max(min(rgb, 1), 0);
rgb255 = round(rgb * 255);
valid_names = csv_names(valid);
valid_L = L(valid);
valid_a = a(valid);
valid_b = b(valid);

n = size(rgb, 1);
if n == 0
    error('没有可显示的有效 L*a*b* 数据。');
end

cols = ceil(sqrt(n));
rows = ceil(n / cols);
cell_h = 30;
cell_w = 30;
pad = 2;
canvas_h = rows * (cell_h + pad) + pad;
canvas_w = cols * (cell_w + pad) + pad;
canvas = ones(canvas_h, canvas_w, 3);
name_grid = strings(rows, cols);
L_grid = nan(rows, cols);
a_grid = nan(rows, cols);
b_grid = nan(rows, cols);
rgb255_grid = nan(rows, cols, 3);

for k = 1:n
    r = floor((k - 1) / cols) + 1;
    c = mod(k - 1, cols) + 1;
    y1 = pad + (r - 1) * (cell_h + pad) + 1;
    y2 = y1 + cell_h - 1;
    x1 = pad + (c - 1) * (cell_w + pad) + 1;
    x2 = x1 + cell_w - 1;
    canvas(y1:y2, x1:x2, 1) = rgb(k, 1);
    canvas(y1:y2, x1:x2, 2) = rgb(k, 2);
    canvas(y1:y2, x1:x2, 3) = rgb(k, 3);
    name_grid(r, c) = valid_names(k);
    L_grid(r, c) = valid_L(k);
    a_grid(r, c) = valid_a(k);
    b_grid(r, c) = valid_b(k);
    rgb255_grid(r, c, :) = rgb255(k, :);
end

fig = figure('Name', 'all_csv_data RGB 总览', 'NumberTitle', 'off');
ax = axes('Parent', fig);
image(ax, canvas);
axis(ax, 'image');
axis(ax, 'off');
title(ax, sprintf('all_csv_data.csv RGB 总览（有效样本 %d / 总样本 %d）', n, height(T)));

info_text = annotation(fig, 'textbox', [0.02 0.01 0.96 0.07], ...
    'String', '移动鼠标到颜色块上可查看 csv_name、L*a*b* 和 RGB', ...
    'FitBoxToText', 'off', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle', ...
    'LineStyle', 'none', ...
    'Color', [0 0 0], ...
    'FontSize', 10);

app.rows = rows;
app.cols = cols;
app.pad = pad;
app.cell_h = cell_h;
app.cell_w = cell_w;
app.name_grid = name_grid;
app.L_grid = L_grid;
app.a_grid = a_grid;
app.b_grid = b_grid;
app.rgb255_grid = rgb255_grid;
app.info_text = info_text;
app.ax = ax;
setappdata(fig, 'rgb_grid_info', app);
set(fig, 'WindowButtonMotionFcn', @local_hover_show_name);

saveas(fig, save_path, 'tiff');

result = struct();
result.csv_path = csv_path;
result.save_path = save_path;
result.sample_count = height(T);
result.valid_count = n;
result.rgb = rgb;
result.rgb255 = rgb255;
result.csv_names = valid_names;

fprintf('RGB 总图已保存: %s\n', save_path);
end

function local_hover_show_name(fig, ~)
app = getappdata(fig, 'rgb_grid_info');
if isempty(app) || ~isfield(app, 'ax') || ~isgraphics(app.ax)
    return;
end
cp = get(app.ax, 'CurrentPoint');
x = cp(1, 1);
y = cp(1, 2);

col = floor((x - app.pad - 1) / (app.cell_w + app.pad)) + 1;
row = floor((y - app.pad - 1) / (app.cell_h + app.pad)) + 1;

if row >= 1 && row <= app.rows && col >= 1 && col <= app.cols
    x0 = app.pad + (col - 1) * (app.cell_w + app.pad) + 1;
    x1 = x0 + app.cell_w - 1;
    y0 = app.pad + (row - 1) * (app.cell_h + app.pad) + 1;
    y1 = y0 + app.cell_h - 1;
    if x >= x0 && x <= x1 && y >= y0 && y <= y1
        name = app.name_grid(row, col);
        if strlength(name) > 0
            L = app.L_grid(row, col);
            a = app.a_grid(row, col);
            b = app.b_grid(row, col);
            rgb255 = squeeze(app.rgb255_grid(row, col, :));
            app.info_text.String = sprintf('csv_name: %s | L*=%.2f | a*=%.2f | b*=%.2f | RGB=(%d, %d, %d)', ...
                name, L, a, b, round(rgb255(1)), round(rgb255(2)), round(rgb255(3)));
            return;
        end
    end
end

app.info_text.String = '移动鼠标到颜色块上可查看 csv_name、L*a*b* 和 RGB';
end
