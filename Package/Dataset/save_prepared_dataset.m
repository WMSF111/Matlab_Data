function dataset = save_prepared_dataset(dataset_tag, X, y, metadata, keep_exports)
% 功能：保存训练数据集
% keep_exports=true  时保存 dataset.mat / X.xlsx / y.xlsx
% keep_exports=false 时仅保存临时 dataset.mat

if nargin < 4 || isempty(metadata)
    metadata = struct();
end
if nargin < 5 || isempty(keep_exports)
    keep_exports = true;
end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if keep_exports
    dataset_dir = fullfile(project_root, 'Result', 'Dataset', dataset_tag);
else
    dataset_dir = fullfile(project_root, 'Result', 'Temp', 'Dataset', dataset_tag);
end
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
dataset.metadata.keep_exports = keep_exports;
dataset.paths = struct();
dataset.paths.dir = dataset_dir;
dataset.paths.mat = fullfile(dataset_dir, 'dataset.mat');
dataset.paths.xlsx_X = '';
dataset.paths.xlsx_y = '';

save(dataset.paths.mat, 'dataset');

if keep_exports
    dataset.paths.xlsx_X = fullfile(dataset_dir, 'X.xlsx');
    dataset.paths.xlsx_y = fullfile(dataset_dir, 'y.xlsx');
    xlswrite(dataset.paths.xlsx_X, X);
    xlswrite(dataset.paths.xlsx_y, dataset.y);

    if isfield(metadata, 'selected_idx') && ~isempty(metadata.selected_idx)
        dataset.paths.xlsx_selected_idx = fullfile(dataset_dir, 'selected_idx.xlsx');
        xlswrite(dataset.paths.xlsx_selected_idx, metadata.selected_idx(:));
    end
    save(dataset.paths.mat, 'dataset');
end
end
