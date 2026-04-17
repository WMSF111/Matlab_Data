function result = compare_e_nose_time_series(temp_list, feature_modes, data_root, lang_mode, normalize_mode, use_baseline_removed)
% Compare E-nose time series across temperatures.

if nargin < 1 || isempty(temp_list)
    temp_list = [70 80 90];
end
if nargin < 2 || isempty(feature_modes)
    feature_modes = {'mean'};
end
if nargin < 3 || isempty(data_root)
    data_root = [];
end
if nargin < 4 || isempty(lang_mode)
    lang_mode = 'zh';
end
if nargin < 5 || isempty(normalize_mode)
    normalize_mode = 'none';
end
if nargin < 6 || isempty(use_baseline_removed)
    use_baseline_removed = true;
end

temp_list = temp_list(:)';
labels = local_get_labels(lang_mode);
[~, norm_info_ref] = normalize_plot_series([0; 1], normalize_mode, [0; 1]);
data_list = cell(1, numel(temp_list));
for i = 1:numel(temp_list)
    data_list{i} = load_e_nose_time_data(temp_list(i), feature_modes, data_root, use_baseline_removed);
end

feature_labels = data_list{1}.feature_labels;
nvar = numel(feature_labels);
ncols = min(4, max(1, ceil(sqrt(nvar))));
nrows = ceil(nvar / ncols);
temp_colors = lines(max(3, numel(temp_list)));

fig = figure('Name', 'E_nose_Temperature_Compare', 'NumberTitle', 'off', 'Color', 'w');
tiledlayout(fig, nrows, ncols, 'Padding', 'compact', 'TileSpacing', 'compact');
plot_data = repmat(local_empty_plot_meta(), nvar, 1);
result_table = table((0:20)', 'VariableNames', {'time_min'});

for i = 1:nvar
    ax = nexttile;
    hold(ax, 'on');
    raw_all_handles = gobjects(0);
    raw_avg_handles = gobjects(0);
    slope_all_handles = gobjects(0);
    slope_avg_handles = gobjects(0);
    raw_all_items = {};
    raw_avg_items = {};
    slope_all_items = {};
    slope_avg_items = {};

    all_vals = [];
    for ti = 1:numel(temp_list)
        S = data_list{ti};
        group_list = S.group_list(:)';
        group_safe_list = S.group_safe_list(:)';
        for bi = 1:numel(group_list)
            group_name = group_list{bi};
            group_safe = group_safe_list{bi};
            idx = strcmp(S.group_label, group_name);
            x_obs = S.minute(idx);
            y_obs_raw = S.feature_matrix(idx, i);
            [x_obs, sort_idx] = sort(x_obs(:));
            y_obs_raw = y_obs_raw(sort_idx);
            if isempty(y_obs_raw)
                continue;
            end
            [y_obs_norm, norm_info] = normalize_plot_series(y_obs_raw, normalize_mode, y_obs_raw);
            all_vals = [all_vals; y_obs_raw(:)]; %#ok<AGROW>
            line_color = temp_colors(ti, :);
            h_raw = plot(ax, x_obs, y_obs_norm, '-o', ...
                'Color', line_color, 'LineWidth', 1.0, 'MarkerSize', 4, ...
                'DisplayName', sprintf(labels.temp_group_fmt, temp_list(ti), group_name));
            raw_all_handles(end+1) = h_raw; %#ok<AGROW>
            raw_all_items{end+1} = sprintf(labels.temp_group_fmt, temp_list(ti), group_name); %#ok<AGROW>

            [sx, sy] = local_compute_slope(x_obs, y_obs_raw);
            h_slope = plot(ax, sx, sy, '--', ...
                'Color', line_color, 'LineWidth', 1.0, 'Visible', 'off', ...
                'DisplayName', sprintf(labels.temp_group_slope_fmt, temp_list(ti), group_name));
            slope_all_handles(end+1) = h_slope; %#ok<AGROW>
            slope_all_items{end+1} = sprintf(labels.temp_group_slope_fmt, temp_list(ti), group_name); %#ok<AGROW>

            result_table.(sprintf('%s_%dC_%s', matlab.lang.makeValidName(feature_labels{i}), temp_list(ti), matlab.lang.makeValidName(group_safe))) = ...
                local_map_to_grid(x_obs, y_obs_raw, S.minute_grid);
        end

        mean_raw = nan(numel(S.minute_grid), 1);
        for mi = 1:numel(S.minute_grid)
            idxm = S.minute == S.minute_grid(mi);
            vals = S.feature_matrix(idxm, i);
            mean_raw(mi) = mean(vals, 'omitnan');
        end
        valid_mean = isfinite(mean_raw);
        mean_norm = normalize_plot_series(mean_raw(valid_mean), normalize_mode, mean_raw(valid_mean));
        h_avg = plot(ax, S.minute_grid(valid_mean), mean_norm, '-', ...
            'Color', temp_colors(ti, :), ...
            'LineWidth', 2.0, ...
            'Visible', 'off', ...
            'DisplayName', sprintf(labels.temp_mean_fmt, temp_list(ti)));
        raw_avg_handles(end+1) = h_avg; %#ok<AGROW>
        raw_avg_items{end+1} = sprintf(labels.temp_mean_fmt, temp_list(ti)); %#ok<AGROW>

        [sx_avg, sy_avg] = local_compute_slope(S.minute_grid(valid_mean), mean_raw(valid_mean));
        h_savg = plot(ax, sx_avg, sy_avg, '-', ...
            'Color', temp_colors(ti, :), ...
            'LineWidth', 2.4, ...
            'Visible', 'off', ...
            'DisplayName', sprintf(labels.temp_mean_slope_fmt, temp_list(ti)));
        slope_avg_handles(end+1) = h_savg; %#ok<AGROW>
        slope_avg_items{end+1} = sprintf(labels.temp_mean_slope_fmt, temp_list(ti)); %#ok<AGROW>

        result_table.(sprintf('%s_%dC_mean', matlab.lang.makeValidName(feature_labels{i}), temp_list(ti))) = mean_raw;
    end

    if isempty(all_vals)
        raw_min = NaN;
        raw_max = NaN;
    else
        raw_min = min(all_vals);
        raw_max = max(all_vals);
    end
    title(ax, sprintf(labels.title_fmt, feature_labels{i}, raw_min, raw_max), 'Interpreter', 'none');
    xlabel(ax, labels.x_label);
    ylabel(ax, sprintf('%s%s', feature_labels{i}, local_get_y_suffix(norm_info_ref, lang_mode)), 'Interpreter', 'none');
    grid(ax, 'on');
    xlim(ax, [0 20]);

    plot_data(i).ax = ax;
    plot_data(i).raw_all_handles = raw_all_handles;
    plot_data(i).raw_avg_handles = raw_avg_handles;
    plot_data(i).slope_all_handles = slope_all_handles;
    plot_data(i).slope_avg_handles = slope_avg_handles;
    plot_data(i).raw_all_items = raw_all_items;
    plot_data(i).raw_avg_items = raw_avg_items;
    plot_data(i).slope_all_items = slope_all_items;
    plot_data(i).slope_avg_items = slope_avg_items;
    plot_data(i).legend_handle = legend(ax, raw_all_handles, raw_all_items, 'Location', 'best');
    set(plot_data(i).legend_handle, 'Visible', 'off');
end

setappdata(fig, 'e_nose_plot_data', plot_data);
setappdata(fig, 'e_nose_labels', labels);
local_add_controls(fig, labels);
local_apply_modes(fig);

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
out_dir = fullfile(project_root, 'Result', 'Summary', 'E_nose_Temperature_Compare');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
writetable(result_table, fullfile(out_dir, 'e_nose_temperature_compare.csv'));
saveas(fig, fullfile(out_dir, 'e_nose_temperature_compare.tif'));

result = struct();
result.figure = fig;
result.output_dir = out_dir;
result.data = data_list;
result.table = result_table;
end

function local_add_controls(fig, labels)
uicontrol(fig, 'Style', 'text', 'String', labels.display_label, ...
    'Units', 'normalized', 'Position', [0.54 0.965 0.08 0.025], ...
    'BackgroundColor', get(fig, 'Color'));
display_popup = uicontrol(fig, 'Style', 'popupmenu', ...
    'String', {labels.display_all, labels.display_avg, labels.display_both}, ...
    'Units', 'normalized', 'Position', [0.62 0.962 0.10 0.03], 'Value', 1, ...
    'Callback', @(~,~) local_apply_modes(fig));

uicontrol(fig, 'Style', 'text', 'String', labels.data_label, ...
    'Units', 'normalized', 'Position', [0.73 0.965 0.06 0.025], ...
    'BackgroundColor', get(fig, 'Color'));
data_popup = uicontrol(fig, 'Style', 'popupmenu', ...
    'String', {labels.data_raw, labels.data_slope}, ...
    'Units', 'normalized', 'Position', [0.79 0.962 0.08 0.03], 'Value', 1, ...
    'Callback', @(~,~) local_apply_modes(fig));

legend_btn = uicontrol(fig, 'Style', 'togglebutton', ...
    'String', labels.show_labels, ...
    'Units', 'normalized', 'Position', [0.88 0.962 0.10 0.03], ...
    'Value', 0, ...
    'Callback', @(src,~) local_toggle_legend_button(src, fig, labels));

setappdata(fig, 'display_popup', display_popup);
setappdata(fig, 'data_popup', data_popup);
setappdata(fig, 'legend_button', legend_btn);
end

function local_toggle_legend_button(src, fig, labels)
if get(src, 'Value') == 1
    set(src, 'String', labels.hide_labels);
else
    set(src, 'String', labels.show_labels);
end
local_apply_modes(fig);
end

function local_apply_modes(fig)
plot_data = getappdata(fig, 'e_nose_plot_data');
display_popup = getappdata(fig, 'display_popup');
data_popup = getappdata(fig, 'data_popup');
legend_btn = getappdata(fig, 'legend_button');

display_id = get(display_popup, 'Value');
data_id = get(data_popup, 'Value');
show_legend = logical(get(legend_btn, 'Value'));

for i = 1:numel(plot_data)
    pd = plot_data(i);
    local_set_visible(pd.raw_all_handles, false);
    local_set_visible(pd.raw_avg_handles, false);
    local_set_visible(pd.slope_all_handles, false);
    local_set_visible(pd.slope_avg_handles, false);

    if data_id == 1
        all_h = pd.raw_all_handles;
        avg_h = pd.raw_avg_handles;
        all_items = pd.raw_all_items;
        avg_items = pd.raw_avg_items;
    else
        all_h = pd.slope_all_handles;
        avg_h = pd.slope_avg_handles;
        all_items = pd.slope_all_items;
        avg_items = pd.slope_avg_items;
    end

    switch display_id
        case 1
            local_set_visible(all_h, true);
            legend_handles = all_h(isgraphics(all_h));
            legend_items = all_items;
        case 2
            local_set_visible(avg_h, true);
            legend_handles = avg_h(isgraphics(avg_h));
            legend_items = avg_items;
        otherwise
            local_set_visible(all_h, true);
            local_set_visible(avg_h, true);
            legend_handles = [all_h(isgraphics(all_h)), avg_h(isgraphics(avg_h))];
            legend_items = [all_items, avg_items];
    end

    if show_legend && ~isempty(legend_handles)
        lgd = legend(pd.ax, legend_handles, legend_items, 'Location', 'best');
        set(lgd, 'Visible', 'on');
        plot_data(i).legend_handle = lgd;
    elseif isgraphics(pd.legend_handle)
        set(pd.legend_handle, 'Visible', 'off');
    end
end

setappdata(fig, 'e_nose_plot_data', plot_data);
end

function local_set_visible(h, flag)
if isempty(h)
    return;
end
valid = isgraphics(h);
if any(valid)
    set(h(valid), 'Visible', local_onoff(flag));
end
end

function txt = local_onoff(flag)
if flag
    txt = 'on';
else
    txt = 'off';
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
valid = isfinite(x) & isfinite(y);
x = x(valid);
y = y(valid);
if numel(x) < 2
    sx = [];
    sy = [];
    return;
end
dx = diff(x);
dy = diff(y);
sx = x(1:end-1) + dx ./ 2;
sy = dy ./ dx;
end

function labels = local_get_labels(lang_mode)
is_zh = strcmpi(lang_mode, 'zh');
labels = struct();
if is_zh
    labels.display_label = char([26174 31034 26041 24335]);
    labels.display_all = char([20840 37096 32452 21035]);
    labels.display_avg = char([24179 22343 20540]);
    labels.display_both = char([20840 37096 32 43 32 24179 22343 20540]);
    labels.data_label = char([25968 25454]);
    labels.data_raw = char([21407 22987 25968 25454]);
    labels.data_slope = char([26012 29575]);
    labels.show_labels = char([26174 31034 26631 31614]);
    labels.hide_labels = char([38544 34255 26631 31614]);
    labels.x_label = char([26102 38388 47 109 105 110]);
    labels.title_fmt = '%s | 最小值=%.4f 最大值=%.4f';
    labels.temp_group_fmt = '%d摄氏度-%s';
    labels.temp_group_slope_fmt = '%d摄氏度-%s斜率';
    labels.temp_mean_fmt = '%d摄氏度平均值';
    labels.temp_mean_slope_fmt = '%d摄氏度平均值斜率';
else
    labels.display_label = 'Display';
    labels.display_all = 'All Groups';
    labels.display_avg = 'Mean';
    labels.display_both = 'All + Mean';
    labels.data_label = 'Mode';
    labels.data_raw = 'Raw';
    labels.data_slope = 'Slope';
    labels.show_labels = 'Show Legend';
    labels.hide_labels = 'Hide Legend';
    labels.x_label = 'Time / min';
    labels.title_fmt = '%s | Min=%.4f Max=%.4f';
    labels.temp_group_fmt = '%dC-%s';
    labels.temp_group_slope_fmt = '%dC-%s Slope';
    labels.temp_mean_fmt = '%dC Mean';
    labels.temp_mean_slope_fmt = '%dC Mean Slope';
end
end

function suffix = local_get_y_suffix(norm_info, lang_mode)
if strcmpi(lang_mode, 'zh')
    suffix = norm_info.y_label_suffix_zh;
else
    suffix = norm_info.y_label_suffix_en;
end
end

function meta = local_empty_plot_meta()
meta = struct('ax', [], ...
    'raw_all_handles', gobjects(0), ...
    'raw_avg_handles', gobjects(0), ...
    'slope_all_handles', gobjects(0), ...
    'slope_avg_handles', gobjects(0), ...
    'raw_all_items', {{}}, ...
    'raw_avg_items', {{}}, ...
    'slope_all_items', {{}}, ...
    'slope_avg_items', {{}}, ...
    'legend_handle', []);
end
