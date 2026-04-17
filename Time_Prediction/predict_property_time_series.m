function result = predict_property_time_series(temp_c, minute_grid, source_csv, lang_mode, normalize_mode)
% Plot observed 0-20 minute changes for all numeric properties at one temperature.

if nargin < 1 || isempty(temp_c)
    temp_c = 70;
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
S = load_temperature_time_data(source_csv, temp_c);
T = S.table;
T = T(T.temperature_c == temp_c, :);
if isempty(T)
    error('No rows were found for %dC.', temp_c);
end

value_vars = S.value_vars;
batch_list = unique(T.rep_id, 'sorted');
batch_list = batch_list(:)';
minutes_obs_all = unique(T.minute, 'sorted');
summary_tbl = table(minutes_obs_all, 'VariableNames', {'minute'});

out_dir = fullfile(S.project_root, 'Result', 'Summary', sprintf('%dC_time_prediction', temp_c));
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

nvar = numel(value_vars);
fig = figure('Name', sprintf('%dC_time_prediction', temp_c), ...
    'NumberTitle', 'off');
tiledlayout(ceil((nvar + 1) / 2), 2, 'Padding', 'compact', 'TileSpacing', 'compact');
legend_handles = gobjects(nvar + 1, 1);
raw_all_handles_cell = cell(nvar + 1, 1);
raw_avg_handles_cell = cell(nvar + 1, 1);
slope_all_handles_cell = cell(nvar + 1, 1);
slope_avg_handles_cell = cell(nvar + 1, 1);
raw_all_legend_items_cell = cell(nvar + 1, 1);
raw_avg_legend_items_cell = cell(nvar + 1, 1);
slope_all_legend_items_cell = cell(nvar + 1, 1);
slope_avg_legend_items_cell = cell(nvar + 1, 1);

for i = 1:nvar
    var_name = value_vars{i};
    raw_min = min(T.(var_name), [], 'omitnan');
    raw_max = max(T.(var_name), [], 'omitnan');
    nexttile;
    hold on;
    color_map = lines(numel(batch_list));
    raw_line_handles = gobjects(1, numel(batch_list));
    raw_avg_handle = gobjects(1, 1);
    slope_line_handles = gobjects(1, numel(batch_list));
    slope_avg_handle = gobjects(1, 1);
    legend_items = cell(1, numel(batch_list));
    norm_info = normalize_plot_series(0, normalize_mode, 0);

    all_minutes = unique(T.minute, 'sorted');
    all_mean = zeros(numel(all_minutes), 1);
    for k = 1:numel(all_minutes)
        idx_mean = T.minute == all_minutes(k);
        all_mean(k) = mean(T.(var_name)(idx_mean), 'omitnan');
    end
    [all_mean_norm, norm_info] = normalize_plot_series(all_mean, normalize_mode, all_mean);
    raw_avg_handle = plot(all_minutes, all_mean_norm, '--', 'LineWidth', 2.2, 'Color', [0.85 0.33 0.10]);
    set(raw_avg_handle, 'Visible', 'off');
    [slope_x_avg, slope_y_avg] = local_compute_slope(all_minutes, all_mean);
    slope_avg_handle = plot(slope_x_avg, slope_y_avg, '--', 'LineWidth', 2.2, 'Color', [0.49 0.18 0.56]);
    set(slope_avg_handle, 'Visible', 'off');
    summary_tbl.(sprintf('%s_mean', var_name)) = local_map_to_grid(all_minutes, all_mean_norm, minutes_obs_all);

    for b = 1:numel(batch_list)
        batch_id = batch_list(b);
        Tb = T(T.rep_id == batch_id, :);
        yb = Tb.(var_name);
        [yb_norm, norm_info] = normalize_plot_series(yb, normalize_mode, yb);
        minutes_obs = Tb.minute;

        summary_mean = nan(numel(minutes_obs_all), 1);
        summary_n = zeros(numel(minutes_obs_all), 1);
        [lia, locb] = ismember(minutes_obs, minutes_obs_all);
        summary_mean(locb(lia)) = yb_norm(lia);
        summary_n(locb(lia)) = 1;
        summary_tbl.(sprintf('%s_batch%d', var_name, batch_id)) = summary_mean;
        summary_tbl.(sprintf('%s_batch%d_n', var_name, batch_id)) = summary_n;

        scatter(minutes_obs, yb_norm, 18, color_map(b, :), 'filled');
        raw_line_handles(b) = plot(minutes_obs, yb_norm, '-', 'LineWidth', 1.6, 'Color', color_map(b, :));
        [slope_x, slope_y] = local_compute_slope(minutes_obs, yb);
        slope_line_handles(b) = plot(slope_x, slope_y, '-', 'LineWidth', 1.6, 'Color', color_map(b, :));
        set(slope_line_handles(b), 'Visible', 'off');
        legend_items{b} = sprintf(labels.batch_legend, batch_id);
    end

    hold off;
    xlim([min(minute_grid), max(minute_grid)]);
    xlabel(labels.x_label);
    ylabel([var_name, local_label_suffix(norm_info, lang_mode)], 'Interpreter', 'none');
    title(sprintf(labels.single_title_with_range, temp_c, var_name, raw_min, raw_max), 'Interpreter', 'none');
    valid = isgraphics(raw_line_handles);
    legend_handles(i) = legend(raw_line_handles(valid), legend_items(valid), 'Location', 'best');
    set(legend_handles(i), 'Visible', 'off');
    raw_all_handles_cell{i} = raw_line_handles(valid);
    raw_avg_handles_cell{i} = raw_avg_handle(isgraphics(raw_avg_handle));
    slope_all_handles_cell{i} = slope_line_handles(isgraphics(slope_line_handles));
    slope_avg_handles_cell{i} = slope_avg_handle(isgraphics(slope_avg_handle));
    raw_all_legend_items_cell{i} = legend_items(valid);
    raw_avg_legend_items_cell{i} = {labels.avg_legend};
    slope_all_legend_items_cell{i} = legend_items(valid);
    slope_avg_legend_items_cell{i} = {labels.slope_avg_legend};
end

lab_cols = local_find_lab_columns(T.Properties.VariableNames);
if lab_cols.valid
    nexttile;
    hold on;
    rgb_all_handles = gobjects(1, numel(batch_list));
    rgb_avg_handle = gobjects(1, 1);
    rgb_legend_items = cell(1, numel(batch_list));

    avg_minutes = unique(T.minute, 'sorted');
    avg_L = zeros(numel(avg_minutes), 1);
    avg_a = zeros(numel(avg_minutes), 1);
    avg_b = zeros(numel(avg_minutes), 1);
    for k = 1:numel(avg_minutes)
        idx_avg = T.minute == avg_minutes(k);
        avg_L(k) = mean(T.(lab_cols.L)(idx_avg), 'omitnan');
        avg_a(k) = mean(T.(lab_cols.a)(idx_avg), 'omitnan');
        avg_b(k) = mean(T.(lab_cols.b)(idx_avg), 'omitnan');
    end
    rgb_avg = local_lab_to_rgb(avg_L, avg_a, avg_b);
    rgb_all_ref = local_lab_to_rgb(T.(lab_cols.L), T.(lab_cols.a), T.(lab_cols.b));
    rgb_min = min(rgb_all_ref(:), [], 'omitnan');
    rgb_max = max(rgb_all_ref(:), [], 'omitnan');
    rgb_avg_handle = hggroup('Parent', gca);
    local_draw_rgb_bars(rgb_avg_handle, avg_minutes, 0, rgb_avg);
    set(rgb_avg_handle, 'Visible', 'off');
    summary_tbl.R_mean = local_map_to_grid(avg_minutes, rgb_avg(:, 1), minutes_obs_all);
    summary_tbl.G_mean = local_map_to_grid(avg_minutes, rgb_avg(:, 2), minutes_obs_all);
    summary_tbl.B_mean = local_map_to_grid(avg_minutes, rgb_avg(:, 3), minutes_obs_all);

    for b = 1:numel(batch_list)
        batch_id = batch_list(b);
        Tb = T(T.rep_id == batch_id, :);
        rgb_batch = local_lab_to_rgb(Tb.(lab_cols.L), Tb.(lab_cols.a), Tb.(lab_cols.b));
        rgb_all_handles(b) = hggroup('Parent', gca);
        local_draw_rgb_bars(rgb_all_handles(b), Tb.minute, batch_id, rgb_batch);
        rgb_legend_items{b} = sprintf(labels.batch_legend, batch_id);
        summary_tbl.(sprintf('R_batch%d', batch_id)) = local_map_to_grid(Tb.minute, rgb_batch(:, 1), minutes_obs_all);
        summary_tbl.(sprintf('G_batch%d', batch_id)) = local_map_to_grid(Tb.minute, rgb_batch(:, 2), minutes_obs_all);
        summary_tbl.(sprintf('B_batch%d', batch_id)) = local_map_to_grid(Tb.minute, rgb_batch(:, 3), minutes_obs_all);
    end

    hold off;
    xlim([min(minute_grid), max(minute_grid)]);
    ylim([-1, numel(batch_list) + 1]);
    yticks([0, batch_list]);
    rgb_tick_labels = [{labels.rgb_mean_tick}, arrayfun(@(x) sprintf(labels.batch_legend, x), batch_list, 'UniformOutput', false)];
    yticklabels(rgb_tick_labels);
    xlabel(labels.x_label);
    ylabel(labels.rgb_ylabel);
    title(sprintf(labels.rgb_title_with_range, temp_c, rgb_min, rgb_max), 'Interpreter', 'none');
    legend_handles(nvar + 1) = legend(rgb_all_handles(isgraphics(rgb_all_handles)), rgb_legend_items(isgraphics(rgb_all_handles)), 'Location', 'best');
    set(legend_handles(nvar + 1), 'Visible', 'off');
    raw_all_handles_cell{nvar + 1} = rgb_all_handles(isgraphics(rgb_all_handles));
    raw_avg_handles_cell{nvar + 1} = rgb_avg_handle(isgraphics(rgb_avg_handle));
    slope_all_handles_cell{nvar + 1} = gobjects(0);
    slope_avg_handles_cell{nvar + 1} = gobjects(0);
    raw_all_legend_items_cell{nvar + 1} = rgb_legend_items(isgraphics(rgb_all_handles));
    raw_avg_legend_items_cell{nvar + 1} = {labels.rgb_avg_legend};
    slope_all_legend_items_cell{nvar + 1} = {};
    slope_avg_legend_items_cell{nvar + 1} = {};
end

local_add_legend_toggle(fig, legend_handles, labels);
local_add_control_panels(fig, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
    raw_all_legend_items_cell, raw_avg_legend_items_cell, slope_all_legend_items_cell, slope_avg_legend_items_cell, labels);

fig_path_tif = fullfile(out_dir, sprintf('%dC_time_prediction.tif', temp_c));
fig_path_fig = fullfile(out_dir, sprintf('%dC_time_prediction.fig', temp_c));
summary_csv = fullfile(out_dir, sprintf('%dC_observed_summary.csv', temp_c));

saveas(fig, fig_path_tif, 'tiff');
savefig(fig, fig_path_fig);
writetable(summary_tbl, summary_csv);

result = struct();
result.temperature_c = temp_c;
result.source_csv = S.source_csv;
result.observed_summary = summary_tbl;
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
        labels.single_title = '%dC | %s';
        labels.single_title_with_range = '%dC | %s | Min=%.3f Max=%.3f';
        labels.batch_legend = 'Batch %d';
        labels.show_labels = 'Show Legend';
        labels.hide_labels = 'Hide Legend';
        labels.display_all = 'All Batches';
        labels.display_avg = 'Mean Only';
        labels.display_both = 'All + Mean';
        labels.display_label = 'Display';
        labels.data_raw = 'Raw';
        labels.data_slope = 'Slope';
        labels.data_label = 'Mode';
        labels.avg_legend = 'Mean';
        labels.slope_avg_legend = 'Mean Slope';
        labels.rgb_avg_legend = 'Mean RGB';
        labels.rgb_title = '%dC | RGB';
        labels.rgb_title_with_range = '%dC | RGB | Min=%.3f Max=%.3f';
        labels.rgb_ylabel = 'RGB';
        labels.rgb_mean_tick = 'Mean';
    otherwise
        labels.x_label = char([26102 38388 47 109 105 110]);
        labels.single_title = [char([37 100]), char([25668 27663 24230]), ' | %s'];
        labels.single_title_with_range = [char([37 100]), char([25668 27663 24230]), ' | %s | ', char([26368 23567 20540 61 37 46 51 102 32 26368 22823 20540 61 37 46 51 102])];
        labels.batch_legend = [char([25209 27425]), char([37 100])];
        labels.show_labels = char([26174 31034 26631 31614]);
        labels.hide_labels = char([38544 34255 26631 31614]);
        labels.display_all = char([20840 37096 25209 27425]);
        labels.display_avg = char([20165 26174 31034 24179 22343 20540]);
        labels.display_both = char([20840 37096 43 24179 22343 25968]);
        labels.display_label = char([26174 31034 26041 24335]);
        labels.data_raw = char([21407 22987 25968 25454]);
        labels.data_slope = char([26012 29575]);
        labels.data_label = char([25968 25454 27169 24335]);
        labels.avg_legend = char([24179 22343 20540]);
        labels.slope_avg_legend = char([24179 22343 26012 29575]);
        labels.rgb_avg_legend = char([24179 22343 20540 32 82 71 66]);
        labels.rgb_title = [char([37 100]), char([25668 27663 24230]), ' | RGB'];
        labels.rgb_title_with_range = [char([37 100]), char([25668 27663 24230]), ' | RGB | ', char([26368 23567 20540 61 37 46 51 102 32 26368 22823 20540 61 37 46 51 102])];
        labels.rgb_ylabel = 'RGB';
        labels.rgb_mean_tick = char([24179 22343 25968]);
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
set(data_popup, 'Callback', @(src, ~) local_update_plot_mode(display_popup, data_popup, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
    raw_all_legend_items_cell, raw_avg_legend_items_cell, slope_all_legend_items_cell, slope_avg_legend_items_cell));
set(display_popup, 'Callback', @(src, ~) local_update_plot_mode(display_popup, data_popup, legend_handles, raw_all_handles_cell, raw_avg_handles_cell, slope_all_handles_cell, slope_avg_handles_cell, ...
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

function rgb = local_lab_to_rgb(L, a, b)
lab = [L(:), a(:), b(:)];
rgb = lab2rgb(lab, 'OutputType', 'double');
rgb = max(min(rgb, 1), 0);
end

function cols = local_find_lab_columns(var_names)
cols = struct('valid', false, 'L', '', 'a', '', 'b', '');
cols.L = local_find_one(var_names, {'L*', 'L_', 'Lx', 'L'});
cols.a = local_find_one(var_names, {'a*', 'a_', 'ax', 'a'});
cols.b = local_find_one(var_names, {'b*', 'b_', 'bx', 'b'});
cols.valid = ~(isempty(cols.L) || isempty(cols.a) || isempty(cols.b));
end

function name = local_find_one(var_names, candidates)
name = '';
for i = 1:numel(candidates)
    idx = find(strcmp(var_names, candidates{i}), 1, 'first');
    if ~isempty(idx)
        name = var_names{idx};
        return;
    end
end
end

function local_draw_rgb_bars(parent_group, x_vals, y_center, rgb_vals)
bar_half_width = 0.5;
bar_half_height = 0.42;
for ii = 1:numel(x_vals)
    x_box = [x_vals(ii) - bar_half_width, x_vals(ii) + bar_half_width, ...
        x_vals(ii) + bar_half_width, x_vals(ii) - bar_half_width];
    y_box = [y_center - bar_half_height, y_center - bar_half_height, ...
        y_center + bar_half_height, y_center + bar_half_height];
    patch('XData', x_box, ...
        'YData', y_box, ...
        'FaceColor', rgb_vals(ii, :), ...
        'EdgeColor', 'none', ...
        'LineWidth', 0.1, ...
        'Parent', parent_group);
end
end
