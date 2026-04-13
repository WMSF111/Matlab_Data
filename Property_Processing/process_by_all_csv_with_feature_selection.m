% 功能：按 all_csv_data.csv 列头建模（先特征筛选，再建模）
% 示例：
%   process_by_all_csv_with_feature_selection('a*')
%   process_by_all_csv_with_feature_selection('a*','sg+msc+snv',3,15,'corr_topk',120)
%   process_by_all_csv_with_feature_selection('a*','sg+msc',3,15,'cars',30)
function process_by_all_csv_with_feature_selection(property_name, preproc_mode, sg_order, sg_window, fs_method, fs_param)

if nargin < 1 || isempty(property_name)
    error('property_name 必填，例如：''a*''。');
end
if nargin < 2 || isempty(preproc_mode), preproc_mode = 'sg+msc'; end
if nargin < 3 || isempty(sg_order), sg_order = 3; end
if nargin < 4 || isempty(sg_window), sg_window = 15; end
if nargin < 5 || isempty(fs_method), fs_method = 'corr_topk'; end
if nargin < 6, fs_param = []; end

project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'Package', 'Training'), '-begin');
addpath(genpath(fullfile(project_root, 'Package')));

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

[y, ~] = build_property_vector_from_all_csv('data\physical\all_csv_data.csv', property_name, 'data\NIR', 1);
sample_count = numel(y);

safe_tag = regexprep(char(property_name), '[^a-zA-Z0-9_]', '_');
time_tag = datestr(now, 'yyyymmdd_HHMMSS');
tag_fs = fullfile(safe_tag, ['FS_', upper(fs_method), '_', time_tag]);

pre_processing_common(tag_fs, sg_order, sg_window, sample_count, preproc_mode);
smooth_path = fullfile(project_root, 'Result', 'Smooth', tag_fs, 'Smooth_Results.mat');
S = load(smooth_path, 'Post_smooth_data');
X = S.Post_smooth_data(1:sample_count, :);

fs_result = select_features_by_method(X, y, fs_method, fs_param);
selected_idx = fs_result.selected_idx;
X_sel = X(:, selected_idx);

fs_dir = fullfile(project_root, 'Result', 'Feature_Select', tag_fs);
if ~exist(fs_dir, 'dir'), mkdir(fs_dir); end
save(fullfile(fs_dir, 'feature_select_result.mat'), 'fs_result');

fig_fs = figure(101);
plot(fs_result.score, 'LineWidth', 1.2); hold on;
scatter(selected_idx, fs_result.score(selected_idx), 12, 'r', 'filled');
title(sprintf('Feature Selection (%s) - %s', upper(fs_method), property_name));
xlabel('Feature Index'); ylabel('Feature Score');
saveas(fig_fs, fullfile(fs_dir, 'feature_select_score.tif'), 'tiff');
close(fig_fs);

[select, not_select] = KS(X_sel, round(sample_count * 0.75));
Xtrain = X_sel(select, :); Ytrain = y(select, :);
Xtest = X_sel(not_select, :); Ytest = y(not_select, :);

CV = plscv(Xtrain, Ytrain, 256, 10, 'center');
A = CV.optLV;
PLS = pls(Xtrain, Ytrain, A, 'center');
Ypred = plsval(PLS, Xtest, Ytest);
Ypred_train = plsval(PLS, Xtrain, Ytrain);

SST_train = sum((Ytrain - mean(Ytrain)).^2);
SSE_train = sum((Ytrain - Ypred_train).^2);
SST_test = sum((Ytest - mean(Ytest)).^2);
SSE_test = sum((Ytest - Ypred).^2);
R2_C = 1 - SSE_train / SST_train;
R2_P = 1 - SSE_test / SST_test;
RMSEC = sqrt(SSE_train / size(Xtrain, 1));
RMSEP = sqrt(SSE_test / size(Xtest, 1));
RPD = std(Ytest) / RMSEP;

model_result = struct();
model_result.property_name = property_name;
model_result.preproc_mode = preproc_mode;
model_result.fs_method = fs_method;
model_result.fs_param = fs_param;
model_result.selected_idx = selected_idx;
model_result.R2_C = R2_C;
model_result.R2_P = R2_P;
model_result.RMSEC = RMSEC;
model_result.RMSEP = RMSEP;
model_result.RPD = RPD;
model_result.Ytest = Ytest;
model_result.Ypred = Ypred;
model_result.Ytrain = Ytrain;
model_result.Ypred_train = Ypred_train;

model_dir = fullfile(project_root, 'Result', 'Model', tag_fs, 'PLS_FS');
if ~exist(model_dir, 'dir'), mkdir(model_dir); end
save(fullfile(model_dir, sprintf('FS_Results_R2_P=%.4f.mat', R2_P)), 'model_result');

fig_model = figure(102);
hold on;
plot(Ytest, Ypred, '.', 'MarkerSize', 15);
plot(Ytrain, Ypred_train, '.', 'MarkerSize', 15);
plot(min(Ytest):max(Ytest), min(Ytest):max(Ytest), 'r-', 'LineWidth', 1.5);
hold off;
xlabel('Actual Values'); ylabel('Predicted Values');
title(sprintf('FS-PLS | R2C=%.3f, R2P=%.3f, RMSEP=%.3f', R2_C, R2_P, RMSEP));
legend('Test Data', 'Train Data', 'Ideal Line', 'Location', 'NorthWest');
saveas(fig_model, fullfile(model_dir, sprintf('FS_Results_R2_P=%.4f.tif', R2_P)), 'tiff');
close(fig_model);

fprintf('特征筛选+建模完成: 属性=%s, 方法=%s, 特征数=%d, R2P=%.4f\n', property_name, fs_method, numel(selected_idx), R2_P);
fprintf('本次结果目录：Result\\Feature_Select\\%s  和  Result\\Model\\%s\\\n', tag_fs, tag_fs);
end


