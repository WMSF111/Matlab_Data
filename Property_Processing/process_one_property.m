% 功能：单属性通用处理入口（旧标签文件模式）
function process_one_property(smooth_tag, physical_file_name, method_name, sample_count)

if nargin < 1 || isempty(smooth_tag), smooth_tag = 'N'; end
if nargin < 2 || isempty(physical_file_name), physical_file_name = 'yingdu.xlsx'; end
if nargin < 3 || isempty(method_name), method_name = 'spa'; end
if nargin < 4 || isempty(sample_count), sample_count = 120; end

project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'Package', 'Training'), '-begin');
addpath(genpath(fullfile(project_root, 'Package')));

black_dir = fullfile(project_root, 'data', 'Black white');
if exist(fullfile(black_dir, 'black.csv'), 'file'), bw='black.csv'; else, c=dir(fullfile(black_dir,'*.csv')); bw=c(1).name; end
Black_White_Processing(fullfile(project_root,'data','NIR'), fullfile(black_dir,bw), fullfile(project_root,'Result','Black_White','post_processing_data.xlsx'));

pre_processing_common(smooth_tag, 3, 15, sample_count);

switch lower(method_name)
    case 'spa'
        pls_spa_common(smooth_tag, physical_file_name, sample_count, 90, 1, inf);
    case 'cars'
        pls_cars_common(smooth_tag, physical_file_name, sample_count, 90, 1000);
    case 'pca'
        pls_pca_common(smooth_tag, physical_file_name, sample_count, 90);
    case 'rfe'
        pls_rfe_common(smooth_tag, physical_file_name, sample_count, 90, 40);
    otherwise
        error('Unsupported method_name: %s. Use "spa", "cars", "pca" or "rfe".', method_name);
end
end
