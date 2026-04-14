% 功能：列出 all_csv_data.csv 中可用的属性列头
% 说明：自动排除第一列 csv_name，仅展示可建模属性列。
% 输出：headers（列头 cell 数组）
function headers = list_all_csv_headers()

project_root = fileparts(fileparts(mfilename('fullpath')));
all_csv_path = fullfile(project_root, 'data', 'physical', 'all_csv_data.csv');
T = readtable(all_csv_path, 'VariableNamingRule', 'preserve');
headers = T.Properties.VariableNames;
headers = headers(~strcmp(headers, 'csv_name'));
disp('Available property headers:');
disp(headers(:));
end
