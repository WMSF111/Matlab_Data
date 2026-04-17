function S = load_temperature_time_data(source_csv, temp_list)
% Load time-series data grouped by temperature from all_csv_data_temp.csv.

if nargin < 1 || isempty(source_csv)
    project_root = fileparts(fileparts(mfilename('fullpath')));
    source_csv = fullfile(project_root, 'data', 'physical', 'all_csv_data_temp.csv');
else
    project_root = fileparts(fileparts(mfilename('fullpath')));
end
if nargin < 2 || isempty(temp_list)
    temp_list = [70, 80, 90];
end

if ~exist(source_csv, 'file')
    error('Source CSV was not found: %s', source_csv);
end

T = readtable(source_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve');
if ~ismember('csv_name', T.Properties.VariableNames)
    error('The source CSV does not contain csv_name.');
end

name_col = string(T.csv_name);
pat = '^([0-9]+)_([0-9]+)_([0-9]+)\.csv$';
tokens = regexp(cellstr(name_col), pat, 'tokens', 'once');
valid = ~cellfun(@isempty, tokens);
if ~any(valid)
    error('csv_name does not match the expected pattern: temp_min_rep.csv');
end

T = T(valid, :);
tokens = tokens(valid);
n = numel(tokens);
temp_vec = zeros(n, 1);
minute_vec = zeros(n, 1);
rep_vec = zeros(n, 1);
for i = 1:n
    tk = tokens{i};
    temp_vec(i) = str2double(tk{1});
    minute_vec(i) = str2double(tk{2});
    rep_vec(i) = str2double(tk{3});
end

T.temperature_c = temp_vec;
T.minute = minute_vec;
T.rep_id = rep_vec;

value_vars = setdiff(T.Properties.VariableNames, {'csv_name', 'temperature_c', 'minute', 'rep_id'}, 'stable');
value_vars = value_vars(varfun(@isnumeric, T(:, value_vars), 'OutputFormat', 'uniform'));
if isempty(value_vars)
    error('No numeric property columns were found.');
end

S = struct();
S.project_root = project_root;
S.source_csv = source_csv;
S.table = T;
S.value_vars = value_vars;
S.temp_list = temp_list(:)';
end
