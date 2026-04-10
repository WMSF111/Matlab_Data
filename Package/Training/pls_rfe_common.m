% 功能：PLS + RFE 公共建模函数（基于 sequentialfs）
% 说明：先做递归特征选择，再进行 PLS 建模与评估。
% 输入：
%   smooth_tag          - 平滑结果标签（用于结果目录）
%   physical_file_name  - 标签文件名（位于 data\physical 下）
%   sample_count        - 样本数
%   ks_count            - KS 划分训练集样本数量
%   nfeatures           - 保留特征数量（默认 40）
% 输出：
%   optimal_result      - 最优模型结果结构体
function optimal_result = pls_rfe_common(smooth_tag, physical_file_name, sample_count, ks_count, nfeatures)

if nargin < 1 || isempty(smooth_tag), smooth_tag = 'N'; end
if nargin < 2 || isempty(physical_file_name), physical_file_name = 'yingdu.xlsx'; end
if nargin < 3 || isempty(sample_count), sample_count = 120; end
if nargin < 4 || isempty(ks_count), ks_count = 90; end
if nargin < 5 || isempty(nfeatures), nfeatures = 40; end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
smooth_folder = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.mat');
physical_data_folder = fullfile(project_root, 'data', 'physical', physical_file_name);
result_folder = fullfile(project_root, 'Result', 'Model', smooth_tag, 'PLS_RFE');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

S = load(smooth_folder, 'Post_smooth_data');
physical_data = xlsread(physical_data_folder);
if ~isfield(S, 'Post_smooth_data')
    error('平滑结果文件缺少变量 Post_smooth_data: %s', smooth_folder);
end
data = S.Post_smooth_data(1:sample_count, :);

[select, not_select] = KS(data, ks_count);
Xtrain = data(select, :); Ytrain = physical_data(select, :);
Xtest = data(not_select, :); Ytest = physical_data(not_select, :);

fun = @(XTrain, YTrain, XTest, YTest) ...
    sqrt(sum((YTest - plsval(pls(XTrain, YTrain, size(XTrain,2), 'center'), XTest, YTest)).^2) / size(XTest, 1));
opts = statset('Display', 'iter', 'UseParallel', false);

max_keep = min([nfeatures, size(Xtrain, 2)]);
[selectedFeatures, ~] = sequentialfs(fun, Xtrain, Ytrain, ...
    'cv', 10, ...
    'direction', 'forward', ...
    'options', opts, ...
    'nfeatures', max_keep);

wave_select = find(selectedFeatures);
Xtrain_RFE = Xtrain(:, wave_select);
Xtest_RFE = Xtest(:, wave_select);
CV = plscv(Xtrain_RFE, Ytrain, 256, 10, 'center');
A = CV.optLV;

PLS = pls(Xtrain_RFE, Ytrain, A, 'center');
Ypred = plsval(PLS, Xtest_RFE, Ytest);
Ypred_train = plsval(PLS, Xtrain_RFE, Ytrain);

SST_train = sum((Ytrain - mean(Ytrain)).^2);
SSE_train = sum((Ytrain - Ypred_train).^2);
SST_test = sum((Ytest - mean(Ytest)).^2);
SSE_test = sum((Ytest - Ypred).^2);
R2_C = 1 - SSE_train / SST_train;
R2_P = 1 - SSE_test / SST_test;
RMSEC = sqrt(SSE_train / size(Xtrain_RFE, 1));
RMSEP = sqrt(SSE_test / size(Xtest_RFE, 1));
RMSECV = CV.RMSECV_min;
RPD = std(Ytest) / RMSEP;

optimal_result.A = A;
optimal_result.wave_select = wave_select;
optimal_result.Q2_Max = CV.Q2_max;
optimal_result.ypred2 = Ypred;
optimal_result.R2_C = R2_C;
optimal_result.R2_P = R2_P;
optimal_result.RMSEC = RMSEC;
optimal_result.RMSEP = RMSEP;
optimal_result.RMSECV = RMSECV;
optimal_result.ypred2_train = Ypred_train;
optimal_result.Rank2 = select;
optimal_result.Xtrain2 = Xtrain;
optimal_result.Ytrain2 = Ytrain;
optimal_result.Xtest2 = Xtest;
optimal_result.ytest2 = Ytest;
optimal_result.RPD = RPD;

results_filename = sprintf('%s\\Results_R2_P=%.4f_Train.mat', result_folder, optimal_result.R2_P);
save(results_filename, 'optimal_result');

fig = figure(1);
hold on;
plot(Ytest, optimal_result.ypred2, '.', 'MarkerSize', 15);
plot(Ytrain, optimal_result.ypred2_train, '.', 'MarkerSize', 15);
plot(min(Ytest):max(Ytest), min(Ytest):max(Ytest), 'r-', 'LineWidth', 1.5);
hold off;
xlabel('Actual Values'); ylabel('Predicted Values');
title(sprintf('R2C=%.3f,R2P=%.3f,RMSEC=%.3f,RMSEP=%.3f', optimal_result.R2_C, optimal_result.R2_P, optimal_result.RMSEC, optimal_result.RMSEP));
legend('Test Data', 'Train Data', 'Ideal Line', 'Location', 'NorthWest');
saveas(fig, sprintf('%s\\Results_R2_P=%.4f_Train.tif', result_folder, optimal_result.R2_P), 'tiff');
close(fig);
end
