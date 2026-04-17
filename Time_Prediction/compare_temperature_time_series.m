function result = compare_temperature_time_series(temp_list, minute_grid, source_csv, lang_mode, normalize_mode)
% Compare observed 70/80/90C time trends on the same figure for all numeric properties.

if nargin < 1 || isempty(temp_list)
    temp_list = [70, 80, 90];
end
if nargin < 2 || isempty(minute_grid)
    minute_grid = 0:20;
end
if nargin < 3 || isempty(source_csv)
    source_csv = [];
end
if nargin < 4 || isempty(lang_mode)
    lang_mode = 'zh';
end
if nargin < 5 || isempty(normalize_mode)
    normalize_mode = 'minmax';
end

labels = local_get_labels(lang_mode);
S = load_temperature_time_data(source_csv, temp_list);
T = S.table;
value_vars = S.value_vars;
temp_list = temp_list(:)';
batch_list = unique(T.rep_id, 'sorted');

out_dir = fullfile(S.project_root, 'Result', 'Summary', 'Time_Prediction_Compare');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

summary_tbl = table(minute_grid(:), 'VariableNames', {'minute'});
nvar = numel(value_vars);
fig = figure(701);
tiledlayout(ceil(nvar / 2), 2, 'Padding', 'compact', 'TileSpacing', 'compact');
color_map = lines(numel(temp_list) * numel(batch_list));
avg_color_map = lines(numel(temp_list));
legend_handles = gobjects(nvar, 1);
raw_all_handles_cell = cell(nvar, 1);
raw_avg_handles_cell = cell(nvar, 1);
slope_all_handles_cell = cell(nvar, 1);
slope_avg_handles_cell = cell(nvar, 1);
raw_all_legend_items_cell = cell(nvar, 1);
raw_avg_legend_items_cell = cell(nvar, 1);
slope_all_legend_items_cell = cell(nvar, 1);
slope_avg_legend_items_cell = cell(nvar, 1);

for i = 1:nvar
    var_name = value_vars{i};
    raw_min = min(T.(var_name), [], 'omitnan');
    raw_max = max(T.(var_name), [], 'omitnan');
    nexttile;
    hold on;
    raw_line_handles = gobjects(1, numel(temp_list) * numel(batch_list));
    raw_avg_handles = gobjects(1, numel(temp_list));
    slope_line_handles = gobjects(1, numel(temp_list) * numel(batch_list));
    slope_avg_handles = gobjects(1, numel(temp_list));
    legend_items = cell(1, numel(temp_list) * numel(batch_list));
    h_idx = 0;
    norm_info = normalize_plot_series(0, normalize_mode, 0);

    for j = 1:numel(temp_list)
        temp_c = temp_list(j);
        temp_block = T(T.temperature_c == temp_c, :);
        avg_minutes = unique(temp_block.minute, 'sorted');
        avg_y = zeros(numel(avg_minutes), 1);
        for k = 1:numel(avg_minutes)
            idx_avg = temp_block.minute == avg_minutes(k);
            avg_y(k) = mean(temp_block.(var_name)(idx_avg), 'omitnan');
        end
        [avg_y_norm, norm_info] = normalize_plot_series(avg_y, normalize_mode, avg_y);
        raw_avg_handles(j) = plot(avg_minutes, avg_y_norm, '--', 'LineWidth', 2.2, 'Color', avg_color_map(j, :));
        set(raw_avg_handles(j), 'Visible', 'off');
        [slope_x_avg, slope_y_avg] = local_compute_slope(avg_minutes, avg_y);
        slope_avg_handles(j) = plot(slope_x_avg, slope_y_avg, '--', 'LineWidth', 2.2, 'Color', avg_color_map(j, :));
        set(slope_avg_handles(j), 'Visible', 'off');
        summary_tbl.(sprintf('%s_%dC_mean', matlab.lang.makeValidName(var_name), temp_c)) = local_map_to_grid(avg_minutes, avg_y_norm, minute_grid(:));

        for b = 1:numel(batch_list)
            batch_id = batch_list(b);
            Tjb = T(T.temperature_c == temp_c & T.rep_id == batch_id, :);
            if isempty(Tjb)
                continue;
            end

            minutes_obs = Tjb.minute;
            yb = Tjb.(var_name);
            [yb_norm, norm_info] = normalize_plot_series(yb, normalize_mode, yb);

            h_idx = h_idx + 1;
            raw_line_handles(h_idx) = plot(minutes_obs, yb_norm, '-', 'LineWidth', 1.6, 'Color', color_map(h_idx, :));
            [slope_x, slope_y] = local_compute_slope(minutes_obs, yb);
            slope_line_handles(h_idx) = plot(slope_x, slope_y, '-', 'LineWidth', 1.6, 'Color', color_map(h_idx, :));
            set(slope_line_handles(h_idx), 'Visible', 'off');
            scatter(minutes_obs, yb_norm, 18, color_map(h_idx, :), 'filled');

            summary_tbl.(sprintf('%s_%dC_batch%d', matlab.lang.makeValidName(var_name), temp_c, batch_id)) = local_map_to_grid(minutes_obs, yb_norm, minute_grid(:));
            legend_items{h_idx} = sprintf(labels.temp_batch_legend, temp_c, batch_id);
        end
    end

    hold off;
    xlim([min(minute_grid), max(minute_grid)]);
    xlabel(labels.x_label);
    ylabel([var_name, local_label_suffix(norm_info, lang_mode)], 'Interpreter', 'none');
    title(sprintf(labels.compare_title_with_range, var_name, raw_min, raw_max), 'Interpreter', 'none');
    valid = isgraphics(raw_line_handles);
    legend_handles(i) = legend(raw_line_handles(valid), legend_items(valid), 'Location', 'best');
    set(legend_handles(i), 'Visible', 'off');
    raw_all_handles_cell{i} = raw_line_handles(valid);
    raw_avg_handles_cell{i} = raw_avg_handles(isgraphics(raw_avg_handles));
    slope_all_handles_cell{i} = slope_line_handles(isgraphics(slope_line_handles));
    slope_avg_handles_cell{i} = slope_avg_handles(isgraphics(slope_avg_handles));
    raw_all_legend_items_cell{i} = legend_items(valid);
    raw_avg_legend_items_cell{i} = arrayfun(@(x) sprintf(labels.avg_legend, x), temp_list, 'UniformOutput', false);
    slope_all_legend_items_cell{i} = legend_items(valid);
    slope_avg_legend_items_cell{i} = arrayfun(@(x) sprintf(labels.slope_avg_legend, x), temp_list, 'UniformOutput', false);
end

local_add_legend_toggle(fig, legend_handles, labels);
local_add_control_panels(fig, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
    raw_all_legend_items_cell, raw_avg_legend_items_cell, slope_all_legend_items_cell, slope_avg_legend_items_cell, labels);

fig_path_tif = fullfile(out_dir, '70_80_90C_time_compare.tif');
fig_path_fig = fullfile(out_dir, '70_80_90C_time_compare.fig');
summary_csv = fullfile(out_dir, '70_80_90C_observed_0_20.csv');

saveas(fig, fig_path_tif, 'tiff');
savefig(fig, fig_path_fig);
writetable(summary_tbl, summary_csv);

result = struct();
result.temp_list = temp_list;
result.source_csv = S.source_csv;
result.observed_table = summary_tbl;
result.output_dir = out_dir;
result.figure_tif = fig_path_tif;
result.figure_fig = fig_path_fig;
result.summary_csv = summary_csv;
result.lang_mode = lang_mode;
result.normalize_mode = normalize_mode;
end

function labels = local_get_labels(lang_mode)
lang_mode = lower(strtrim(string(lang_mode)));
switch char(lang_mode)
    case 'en'
        labels.x_label = 'Time / min';
        labels.temp_batch_legend = '%dC-Batch%d';
        labels.compare_title = 'Temperature Compare | %s';
        labels.compare_title_with_range = 'Temperature Compare | %s | Min=%.3f Max=%.3f';
        labels.show_labels = 'Show Legend';
        labels.hide_labels = 'Hide Legend';
        labels.display_all = 'All Batches';
        labels.display_avg = 'Mean Only';
        labels.display_both = 'All + Mean';
        labels.display_label = 'Display';
        labels.data_raw = 'Raw';
        labels.data_slope = 'Slope';
        labels.data_label = 'Mode';
        labels.avg_legend = '%dC Mean';
        labels.slope_avg_legend = '%dC Mean Slope';
    otherwise
        labels.x_label = char([26102 38388 47 109 105 110]);
        labels.temp_batch_legend = [char([37 100]), char([25668 27663 24230 45 25209 27425]), char([37 100])];
        labels.compare_title = [char([19981 21516 28201 24230 23545 27604 27604]), ' | %s'];
        labels.compare_title_with_range = [char([19981 21516 28201 24230 23545 27604 27604]), ' | %s | ', char([26368 23567 20540 61 37 46 51 102 32 26368 22823 20540 61 37 46 51 102])];
        labels.show_labels = char([26174 31034 26631 31614]);
        labels.hide_labels = char([38544 34255 26631 31614]);
        labels.display_all = char([20840 37096 25209 27425]);
        labels.display_avg = char([20165 26174 31034 24179 22343 20540]);
        labels.display_both = char([20840 37096 43 24179 22343 25968]);
        labels.display_label = char([26174 31034 26041 24335]);
        labels.data_raw = char([21407 22987 25968 25454]);
        labels.data_slope = char([26012 29575]);
        labels.data_label = char([25968 25454 27169 24335]);
        labels.avg_legend = [char([37 100]), char([25668 27663 24230 24179 22343 20540])];
        labels.slope_avg_legend = [char([37 100]), char([25668 27663 24230 24179 22343 26012 29575])];
end
end

function txt = local_label_suffix(info, lang_mode)
if strcmpi(lang_mode, 'en')
    txt = info.y_label_suffix_en;
else
    txt = info.y_label_suffix_zh;
end
end

function local_add_legend_toggle(fig, legend_handles, labels)
btn = uicontrol(fig, 'Style', 'togglebutton', ...
    'String', labels.show_labels, ...
    'Units', 'normalized', ...
    'Position', [0.86 0.95 0.12 0.04], ...
    'Value', 0, ...
    'Callback', @(src, ~) local_toggle_legends(src, legend_handles, labels));
setappdata(fig, 'legend_toggle_button', btn);
end

function local_toggle_legends(src, legend_handles, labels)
show_flag = logical(get(src, 'Value'));
valid = isgraphics(legend_handles);
if show_flag
    set(legend_handles(valid), 'Visible', 'on');
    set(src, 'String', labels.hide_labels);
else
    set(legend_handles(valid), 'Visible', 'off');
    set(src, 'String', labels.show_labels);
end
end

function local_add_control_panels(fig, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
    raw_all_legend_items_cell, raw_avg_legend_items_cell, slope_all_legend_items_cell, slope_avg_legend_items_cell, labels)
uicontrol(fig, 'Style', 'text', ...
    'String', labels.display_label, ...
    'Units', 'normalized', ...
    'Position', [0.60 0.95 0.08 0.03], ...
    'BackgroundColor', get(fig, 'Color'));
display_popup = uicontrol(fig, 'Style', 'popupmenu', ...
    'String', {labels.display_all, labels.display_avg, labels.display_both}, ...
    'Units', 'normalized', ...
    'Position', [0.68 0.95 0.09 0.04], ...
    'Value', 1);
uicontrol(fig, 'Style', 'text', ...
    'String', labels.data_label, ...
    'Units', 'normalized', ...
    'Position', [0.78 0.95 0.07 0.03], ...
    'BackgroundColor', get(fig, 'Color'));
data_popup = uicontrol(fig, 'Style', 'popupmenu', ...
    'String', {labels.data_raw, labels.data_slope}, ...
    'Units', 'normalized', ...
    'Position', [0.85 0.95 0.08 0.04], ...
    'Value', 1);
set(display_popup, 'Callback', @(src, ~) local_update_plot_mode(display_popup, data_popup, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
    raw_all_legend_items_cell, raw_avg_legend_items_cell, slope_all_legend_items_cell, slope_avg_legend_items_cell));
set(data_popup, 'Callback', @(src, ~) local_update_plot_mode(display_popup, data_popup, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
    raw_all_legend_items_cell, raw_avg_legend_items_cell, slope_all_legend_items_cell, slope_avg_legend_items_cell));
local_update_plot_mode(display_popup, data_popup, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
    raw_all_legend_items_cell, raw_avg_legend_items_cell, slope_all_legend_items_cell, slope_avg_legend_items_cell);
end

function local_update_plot_mode(display_popup, data_popup, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
    raw_all_legend_items_cell, raw_avg_legend_items_cell, slope_all_legend_items_cell, slope_avg_legend_items_cell)
display_id = get(display_popup, 'Value');
data_id = get(data_popup, 'Value');
for i = 1:numel(raw_all_handles_cell)
    if data_id == 1
        all_h = raw_all_handles_cell{i};
        avg_h = raw_avg_handles_cell{i};
        all_items = raw_all_legend_items_cell{i};
        avg_items = raw_avg_legend_items_cell{i};
    else
        all_h = slope_all_handles_cell{i};
        avg_h = slope_avg_handles_cell{i};
        all_items = slope_all_legend_items_cell{i};
        avg_items = slope_avg_legend_items_cell{i};
    end
    local_set_handle_visibility(raw_all_handles_cell{i}, false);
    local_set_handle_visibility(raw_avg_handles_cell{i}, false);
    local_set_handle_visibility(slope_all_handles_cell{i}, false);
    local_set_handle_visibility(slope_avg_handles_cell{i}, false);
    if ~isempty(all_h)
        local_set_handle_visibility(all_h, display_id == 1 || display_id == 3);
    end
    if ~isempty(avg_h)
        local_set_handle_visibility(avg_h, display_id == 2 || display_id == 3);
    end
    local_update_legend_content(legend_handles(i), all_h, avg_h, all_items, avg_items, display_id);
end
end

function local_set_handle_visibility(h, flag)
if isempty(h)
    return;
end
valid = isgraphics(h);
if any(valid)
    set(h(valid), 'Visible', ternary_visible(flag));
end
end

function s = ternary_visible(flag)
if flag
    s = 'on';
else
    s = 'off';
end
end

function local_update_legend_content(legend_handle, all_h, avg_h, all_items, avg_items, display_id)
if ~isgraphics(legend_handle)
    return;
end
switch display_id
    case 1
        handles = all_h(isgraphics(all_h));
        items = all_items;
    case 2
        handles = avg_h(isgraphics(avg_h));
        items = avg_items;
    otherwise
        handles = [all_h(isgraphics(all_h)), avg_h(isgraphics(avg_h))];
        items = [all_items, avg_items];
end
if isempty(handles)
    set(legend_handle, 'Visible', 'off');
    return;
end
ax = local_get_legend_axes(legend_handle);
if isempty(ax) || ~isgraphics(ax, 'axes')
    return;
end
was_visible = get(legend_handle, 'Visible');
new_legend = legend(ax, handles, items, 'Location', 'best');
set(new_legend, 'Visible', was_visible);
end

function ax = local_get_legend_axes(legend_handle)
ax = [];
if ~isgraphics(legend_handle)
    return;
end
try
    ax = get(legend_handle, 'PlotAxes');
catch
end
if isempty(ax)
    try
        ax = legend_handle.PlotAxes;
    catch
    end
end
if iscell(ax) && ~isempty(ax)
    ax = ax{1};
end
if isempty(ax) || ~isgraphics(ax)
    try
        ax = ancestor(legend_handle, 'axes');
    catch
        ax = [];
    end
end
end

function out_col = local_map_to_grid(x_obs, y_obs, x_grid)
out_col = nan(numel(x_grid), 1);
[lia, locb] = ismember(x_obs, x_grid);
out_col(locb(lia)) = y_obs(lia);
end

function [sx, sy] = local_compute_slope(x, y)
x = x(:);
y = y(:);
if numel(x) < 2
    sx = x;
    sy = nan(size(x));
    return;
end
dx = diff(x);
dy = diff(y);
sx = x(1:end-1) + dx / 2;
sy = dy ./ dx;
end
