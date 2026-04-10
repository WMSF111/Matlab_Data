% 功能：根据 all_csv_data.csv 构建目标属性向量 Y
% 说明：
%   1) 按 NIR 文件名排序
%   2) 按 average_num 分组（与黑白矫正分组规则一致）
%   3) 对每组属性值取均值，得到建模目标向量
% 输入：
%   all_csv_relative_path - all_csv_data.csv 相对路径
%   property_name         - 目标属性列名（如 PH、Mrix、L*）
%   csv_subfolder         - NIR 数据目录
%   average_num           - 分组平均个数（默认 3）
% 输出：
%   y                     - 目标向量
%   grouped_file_names    - 每组对应的文件名
function [y, grouped_file_names] = build_property_vector_from_all_csv(all_csv_relative_path, property_name, csv_subfolder, average_num)

if nargin < 1 || isempty(all_csv_relative_path), all_csv_relative_path = 'data\physical\all_csv_data.csv'; end
if nargin < 2 || isempty(property_name), error('property_name is required.'); end
if nargin < 3 || isempty(csv_subfolder), csv_subfolder = 'data\NIR'; end
if nargin < 4 || isempty(average_num), average_num = 3; end

project_root = fileparts(fileparts(mfilename('fullpath')));
all_csv_path = fullfile(project_root, strrep(all_csv_relative_path, '\', filesep));
csv_folder = fullfile(project_root, strrep(csv_subfolder, '\', filesep));

T = readtable(all_csv_path, 'VariableNamingRule', 'preserve');
col_names = string(T.Properties.VariableNames);
if ~ismember("csv_name", col_names)
    error('all_csv_data.csv must contain column "csv_name".');
end
if ~ismember(string(property_name), col_names)
    error('Property "%s" not found in all_csv_data.csv.', property_name);
end

csv_name_col = string(T{:, "csv_name"});
prop_col = T{:, property_name};
if iscell(prop_col)
    prop_col = str2double(string(prop_col));
end
prop_col = double(prop_col);

% Build name -> value map
name_to_value = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(csv_name_col)
    name_to_value(char(csv_name_col(i))) = prop_col(i);
end

nir_files = dir(fullfile(csv_folder, '*.csv'));
[~, idx] = sort({nir_files.name});
nir_files = nir_files(idx);
nir_names = string({nir_files.name});

group_count = floor(numel(nir_names) / average_num);
y = zeros(group_count, 1);
grouped_file_names = strings(group_count, average_num);

for g = 1:group_count
    id0 = (g - 1) * average_num + 1;
    ids = id0:(id0 + average_num - 1);
    group_names = nir_names(ids);
    grouped_file_names(g, :) = group_names;

    vals = zeros(average_num, 1);
    for k = 1:average_num
        key = char(group_names(k));
        if ~isKey(name_to_value, key)
            error('csv_name "%s" exists in NIR folder but not in all_csv_data.csv.', key);
        end
        vals(k) = name_to_value(key);
    end
    y(g) = mean(vals, 'omitnan');
end
end
