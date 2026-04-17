function output = merge_summary_csv_by_property(property_name)
% 功能：将 Result\Summary 下同一预测值的 all_results.csv 合并为一个总表
% 用法：
%   output = merge_summary_csv_by_property('C*')
%   output = merge_summary_csv_by_property('a*')

if nargin < 1 || isempty(property_name)
    error('请提供预测对象，例如：''C*''、''a*''。');
end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
summary_root = fullfile(project_root, 'Result', 'Summary');
property_tag = property_to_tag(property_name);
pattern = [property_tag '_regressor_compare_*'];

dirs = dir(fullfile(summary_root, pattern));
dirs = dirs([dirs.isdir]);
if isempty(dirs)
    error('未找到对应预测值的结果目录：%s', pattern);
end

merged_table = table();
source_count = 0;
all_var_names = {};
table_list = cell(0, 1);

for i = 1:numel(dirs)
    csv_path = fullfile(dirs(i).folder, dirs(i).name, 'all_results.csv');
    if ~exist(csv_path, 'file')
        continue;
    end

    T = readtable(csv_path, 'VariableNamingRule', 'preserve');
    T.source_dir = repmat(string(dirs(i).name), height(T), 1);
    T.source_csv = repmat(string(csv_path), height(T), 1);

    table_list{end + 1, 1} = T; %#ok<AGROW>
    all_var_names = union(all_var_names, T.Properties.VariableNames, 'stable');
    source_count = source_count + 1;
end

if isempty(table_list)
    error('找到结果目录，但没有可合并的 all_results.csv。');
end

for i = 1:numel(table_list)
    T = table_list{i};
    missing_names = setdiff(all_var_names, T.Properties.VariableNames, 'stable');
    for j = 1:numel(missing_names)
        if isnumeric_column_name(missing_names{j})
            T.(missing_names{j}) = nan(height(T), 1);
        else
            T.(missing_names{j}) = strings(height(T), 1);
        end
    end
    T = T(:, all_var_names);

    if isempty(merged_table)
        merged_table = T;
    else
        merged_table = [merged_table; T]; %#ok<AGROW>
    end
end

if ismember('R2_P', merged_table.Properties.VariableNames) && ismember('RMSEP', merged_table.Properties.VariableNames)
    merged_table = sortrows(merged_table, {'R2_P', 'RMSEP'}, {'descend', 'ascend'});
end

out_csv = fullfile(summary_root, [property_tag '_all_results_merged.csv']);
out_mat = fullfile(summary_root, [property_tag '_all_results_merged.mat']);

save(out_mat, 'merged_table');
writetable(merged_table, out_csv);

output.property_name = property_name;
output.property_tag = property_tag;
output.source_count = source_count;
output.row_count = height(merged_table);
output.csv_path = out_csv;
output.mat_path = out_mat;

fprintf('已合并 %d 份结果，输出 CSV：%s\n', source_count, out_csv);
fprintf('对应 MAT：%s\n', out_mat);
fprintf('总记录数：%d\n', height(merged_table));
end

function tf = isnumeric_column_name(var_name)
numeric_names = {'R2_C','R2_P','RMSEC','RMSEP','RPD','SG阶数','SG窗口'};
tf = ismember(var_name, numeric_names);
end

function tag = property_to_tag(property_name)
tag = lower(strtrim(property_name));
tag = strrep(tag, '*', '_star');
tag = regexprep(tag, '[^a-zA-Z0-9_]+', '_');
if isempty(tag)
    tag = 'property';
end
end
