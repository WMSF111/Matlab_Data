function S = load_e_nose_time_data(temp_c, feature_modes, data_root, use_baseline_removed)
% Load E-nose time-series features for one temperature.

if nargin < 2 || isempty(feature_modes)
    feature_modes = {'mean'};
end
if nargin < 3 || isempty(data_root)
    project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    data_root = fullfile(project_root, 'data', 'E_nose');
end
if nargin < 4 || isempty(use_baseline_removed)
    use_baseline_removed = true;
end

temp_dir = fullfile(data_root, num2str(temp_c));
if ~isfolder(temp_dir)
    error('E-nose temperature folder not found: %s', temp_dir);
end

dir_info = dir(temp_dir);
dir_info = dir_info([dir_info.isdir]);
dir_names = {dir_info.name};
dir_names = dir_names(~ismember(dir_names, {'.', '..'}));

batch_nums = [];
for i = 1:numel(dir_names)
    tok = regexp(dir_names{i}, '^(\d+)(?:_.*)?$', 'tokens', 'once');
    if ~isempty(tok)
        batch_nums(end+1) = str2double(tok{1}); %#ok<AGROW>
    end
end
batch_list = unique(batch_nums);
batch_list = batch_list(:)';

minutes = 0:20;
records = [];
sensor_names = {};
feature_labels = {};
group_defs = struct('batch', {}, 'parity', {}, 'sub_dir', {}, 'label', {}, 'safe_id', {});

for bi = 1:numel(batch_list)
    batch_id = batch_list(bi);
    batch_dir = local_select_batch_dir(temp_dir, batch_id, use_baseline_removed);
    if isempty(batch_dir)
        continue;
    end
    curr_groups = local_collect_group_defs(batch_dir, batch_id);
    group_defs = [group_defs, curr_groups]; %#ok<AGROW>
    for gi = 1:numel(curr_groups)
        for mi = 1:numel(minutes)
            minute = minutes(mi);
            csv_path = fullfile(curr_groups(gi).sub_dir, sprintf('%d.csv', minute));
            if ~exist(csv_path, 'file')
                continue;
            end
            T = readtable(csv_path, 'TextType', 'string', 'VariableNamingRule', 'preserve');
            if width(T) < 2
                continue;
            end
            curr_sensor_names = T.Properties.VariableNames(2:end);
            data_matrix = table2array(T(:, 2:end));
            [feat_row, feat_labels, feature_modes] = extract_e_nose_features(data_matrix, curr_sensor_names, feature_modes);
            if isempty(sensor_names)
                sensor_names = curr_sensor_names;
                feature_labels = feat_labels;
            end
            rec = struct();
            rec.temp = temp_c;
            rec.batch = batch_id;
            rec.parity = curr_groups(gi).parity;
            rec.group_label = curr_groups(gi).label;
            rec.group_safe_id = curr_groups(gi).safe_id;
            rec.minute = minute;
            rec.csv_path = csv_path;
            rec.feature_row = feat_row;
            records = [records; rec]; %#ok<AGROW>
        end
    end
end

if isempty(records)
    error('No E-nose records found for %dC in %s', temp_c, temp_dir);
end

feature_matrix = vertcat(records.feature_row);

S = struct();
S.temp_c = temp_c;
S.data_root = data_root;
S.use_baseline_removed = use_baseline_removed;
S.feature_modes = feature_modes;
S.sensor_names = sensor_names;
S.feature_labels = feature_labels;
S.feature_matrix = feature_matrix;
S.batch = vertcat(records.batch);
S.parity = {records.parity}';
S.group_label = {records.group_label}';
S.group_safe_id = {records.group_safe_id}';
S.minute = vertcat(records.minute);
S.csv_path = {records.csv_path}';
S.batch_list = unique(S.batch)';
S.group_list = unique(S.group_label, 'stable')';
S.group_safe_list = unique(S.group_safe_id, 'stable')';
S.minute_grid = minutes;
end

function group_defs = local_collect_group_defs(batch_dir, batch_id)
odd_name = char([22855 25968]);
even_name = char([20598 25968]);
group_defs = struct('batch', {}, 'parity', {}, 'sub_dir', {}, 'label', {}, 'safe_id', {});

odd_dir = fullfile(batch_dir, odd_name);
even_dir = fullfile(batch_dir, even_name);

if isfolder(odd_dir)
    group_defs(end+1) = struct( ... %#ok<AGROW>
        'batch', batch_id, ...
        'parity', 'odd', ...
        'sub_dir', odd_dir, ...
        'label', sprintf('%d-%s', batch_id, odd_name), ...
        'safe_id', sprintf('batch%d_odd', batch_id));
end

if isfolder(even_dir)
    group_defs(end+1) = struct( ... %#ok<AGROW>
        'batch', batch_id, ...
        'parity', 'even', ...
        'sub_dir', even_dir, ...
        'label', sprintf('%d-%s', batch_id, even_name), ...
        'safe_id', sprintf('batch%d_even', batch_id));
end

if isempty(group_defs)
    group_defs(end+1) = struct( ... %#ok<AGROW>
        'batch', batch_id, ...
        'parity', 'all', ...
        'sub_dir', batch_dir, ...
        'label', sprintf('%d', batch_id), ...
        'safe_id', sprintf('batch%d', batch_id));
end
end

function batch_dir = local_select_batch_dir(temp_dir, batch_id, use_baseline_removed)
batch_dir = '';
prefer_dir = fullfile(temp_dir, sprintf('%d_%s', batch_id, char([21435 22522])));
raw_dir = fullfile(temp_dir, sprintf('%d', batch_id));

if use_baseline_removed && isfolder(prefer_dir)
    batch_dir = prefer_dir;
elseif isfolder(raw_dir)
    batch_dir = raw_dir;
elseif isfolder(prefer_dir)
    batch_dir = prefer_dir;
end
end
