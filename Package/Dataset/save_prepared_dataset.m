function dataset = save_prepared_dataset(dataset_tag, X, y, metadata)
% 功能：保存解耦后的训练数据集
% 输出目录：Result/Dataset/<dataset_tag>/

if nargin < 4 || isempty(metadata)
    metadata = struct();
end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
dataset_dir = fullfile(project_root, 'Result', 'Dataset', dataset_tag);
if ~exist(dataset_dir, 'dir')
    mkdir(dataset_dir);
end

dataset = struct();
dataset.X = X;
dataset.y = y(:);
dataset.metadata = metadata;
dataset.metadata.dataset_tag = dataset_tag;
dataset.metadata.sample_count = size(X, 1);
dataset.metadata.feature_count = size(X, 2);
dataset.metadata.created_at = datestr(now, 'yyyy-mm-dd HH:MM:SS');
dataset.paths = struct();
dataset.paths.dir = dataset_dir;
dataset.paths.mat = fullfile(dataset_dir, 'dataset.mat');
dataset.paths.xlsx_X = fullfile(dataset_dir, 'X.xlsx');
dataset.paths.xlsx_y = fullfile(dataset_dir, 'y.xlsx');

save(dataset.paths.mat, 'dataset');
xlswrite(dataset.paths.xlsx_X, X);
xlswrite(dataset.paths.xlsx_y, dataset.y);

if isfield(metadata, 'selected_idx') && ~isempty(metadata.selected_idx)
    dataset.paths.xlsx_selected_idx = fullfile(dataset_dir, 'selected_idx.xlsx');
    xlswrite(dataset.paths.xlsx_selected_idx, metadata.selected_idx(:));
    save(dataset.paths.mat, 'dataset');
end
end
