% 功能：按 all_csv_data.csv 的列头进行属性预测
% 说明：
%   1) 根据 csv_name 对齐 NIR 文件
%   2) 按与黑白矫正一致的分组规则构建目标向量
%   3) 进行预处理与建模
% 示例：
%   process_by_all_csv_header('Mrix', 'spa')
%   process_by_all_csv_header('a*', 'cars', 'sg+msc+snv', 3, 15)
function process_by_all_csv_header(property_name, method_name, preproc_mode, sg_order, sg_window)

if nargin < 1 || isempty(property_name)
    error('property_name is required, e.g. "PH", "Mrix", "L*", "a*", "b*", "C*", "ho".');
end
if nargin < 2 || isempty(method_name)
    method_name = 'spa'; % 'spa' / 'cars' / 'pca' / 'rfe'
end
if nargin < 3 || isempty(preproc_mode)
    preproc_mode = 'sg+msc';
end
if nargin < 4 || isempty(sg_order)
    sg_order = 3;
end
if nargin < 5 || isempty(sg_window)
    sg_window = 15;
end
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'Package', 'Training'), '-begin');
addpath(genpath(fullfile(project_root, 'Package')));

% 1) 根据当前近红外文件重建黑白矫正结果（仅保留一个黑白处理函数）
black_dir = fullfile(project_root, 'data', 'Black white');
if exist(fullfile(black_dir, 'black.csv'), 'file')
    black_file_name = 'black.csv';
else
    cands = dir(fullfile(black_dir, '*.csv'));
    if isempty(cands), error('未找到黑白参考文件。'); end
    black_file_name = cands(1).name;
end
csv_folder = fullfile(project_root, 'data', 'NIR');
black_file = fullfile(black_dir, black_file_name);
black_out = fullfile(project_root, 'Result', 'Black_White', 'post_processing_data.xlsx');
if exist(black_out, 'file'), delete(black_out); end
Black_White_Processing(csv_folder, black_file, black_out);

% 2) 使用相同的文件分组规则，将“all_csv_data.csv”文件中的数据构建为“Y”数据集。
[y, ~] = build_property_vector_from_all_csv('data\physical\all_csv_data.csv', property_name, 'data\NIR', 3);
sample_count = numel(y);

% 3) Save temporary target file so existing modeling functions can reuse it
safe_tag = regexprep(char(property_name), '[^a-zA-Z0-9_]', '_');
time_tag = datestr(now, 'yyyymmdd_HHMMSS');
run_tag = fullfile(safe_tag, time_tag); % 结果分组：Result/<属性>/<时间戳>/...

tmp_rel = sprintf('data\\physical\\generated_targets\\%s\\%s\\target_from_all_csv.xlsx', safe_tag, time_tag);
tmp_abs = fullfile(project_root, strrep(tmp_rel, '\', filesep));
tmp_dir = fileparts(tmp_abs);
if ~exist(tmp_dir, 'dir')
    mkdir(tmp_dir);
end
xlswrite(tmp_abs, y);

% 4) Pre-processing and model
pre_processing_common(run_tag, sg_order, sg_window, sample_count, preproc_mode);

switch lower(method_name)
    case 'spa'
        pls_spa_common(run_tag, sprintf('generated_targets\\%s\\%s\\target_from_all_csv.xlsx', safe_tag, time_tag), sample_count, round(sample_count * 0.75), 1, inf);
    case 'cars'
        pls_cars_common(run_tag, sprintf('generated_targets\\%s\\%s\\target_from_all_csv.xlsx', safe_tag, time_tag), sample_count, round(sample_count * 0.75), 1000);
    case 'pca'
        pls_pca_common(run_tag, sprintf('generated_targets\\%s\\%s\\target_from_all_csv.xlsx', safe_tag, time_tag), sample_count, round(sample_count * 0.75));
    case 'rfe'
        pls_rfe_common(run_tag, sprintf('generated_targets\\%s\\%s\\target_from_all_csv.xlsx', safe_tag, time_tag), sample_count, round(sample_count * 0.75), 40);
    otherwise
        error('Unsupported method_name: %s. Use "spa", "cars", "pca" or "rfe".', method_name);
end

fprintf('本次结果目录：Result\\Smooth\\%s  和  Result\\Model\\%s\\\n', run_tag, run_tag);
end
