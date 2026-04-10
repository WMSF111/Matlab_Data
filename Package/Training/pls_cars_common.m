% 功能：PLS + CARS 公共建模函数
% 说明：完成样本划分、CARS 特征筛选、PLS 建模、评估并保存结果。
% 输入：
%   smooth_tag          - 平滑结果标签（用于结果目录）
%   physical_file_name  - 标签文件名（位于 data\physical 下）
%   sample_count        - 样本数
%   ks_count            - KS 划分训练集样本数量
%   num_iterations      - CARS 迭代次数
% 输出：
%   optimal_result      - 最优模型结果结构体
function optimal_result = pls_cars_common(smooth_tag, physical_file_name, sample_count, ks_count, num_iterations)

if nargin < 1 || isempty(smooth_tag), smooth_tag = 'N'; end
if nargin < 2 || isempty(physical_file_name), physical_file_name = 'yingdu.xlsx'; end
if nargin < 3 || isempty(sample_count), sample_count = 120; end
if nargin < 4 || isempty(ks_count), ks_count = 90; end
if nargin < 5 || isempty(num_iterations), num_iterations = 1000; end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
smooth_folder = fullfile(project_root, 'Result', 'Smooth', smooth_tag, 'Smooth_Results.mat');
physical_data_folder = fullfile(project_root, 'data', 'physical', physical_file_name);
result_folder = fullfile(project_root, 'Result', 'Model', smooth_tag, 'PLS_CARS');
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

CV = plscv(Xtrain, Ytrain, 256, 10, 'center');
A = CV.optLV;

max_R2_P = -inf;
Good_RMSE = inf;
optimal_result = struct();

for j = 1:num_iterations
    CARS = carspls(Xtrain, Ytrain, A, 10, 'center', 30, 0, 1);
    excellent_feature = CARS.vsel;
    post_data = data(:, excellent_feature);
    Xtrain_CV = post_data(select, :); Ytrain_CV = physical_data(select, :);
    Xtest_CV = post_data(not_select, :); Ytest_CV = physical_data(not_select, :);

    CV2 = plscv(Xtrain_CV, Ytrain_CV, 256, 10, 'center');
    A2 = CV2.optLV;
    PLS = pls(Xtrain_CV, Ytrain_CV, A2, 'center');
    Ypred = plsval(PLS, Xtest_CV, Ytest_CV);
    Ypred_train = plsval(PLS, Xtrain_CV, Ytrain_CV);

    SST_train = sum((Ytrain_CV - mean(Ytrain_CV)).^2);
    SSE_train = sum((Ytrain_CV - Ypred_train).^2);
    SST_test = sum((Ytest_CV - mean(Ytest_CV)).^2);
    SSE_test = sum((Ytest_CV - Ypred).^2);
    R2_C = 1 - SSE_train / SST_train;
    R2_P = 1 - SSE_test / SST_test;
    RMSEC = sqrt(SSE_train / size(Xtrain_CV, 1));
    RMSEP = sqrt(SSE_test / size(Xtest_CV, 1));
    RMSECV = CV.RMSECV_min;
    RPD = std(Ytest_CV) / RMSEP;

    if R2_P > max_R2_P && (RMSEP - RMSEC) < Good_RMSE
        Good_RMSE = RMSEP - RMSEC;
        max_R2_P = R2_P;
        optimal_result.A = A2;
        optimal_result.Q2_Max = CV2.Q2_max;
        optimal_result.ypred2 = Ypred;
        optimal_result.selected_features = excellent_feature;
        optimal_result.R2_C = R2_C;
        optimal_result.R2_P = R2_P;
        optimal_result.RMSEC = RMSEC;
        optimal_result.RMSEP = RMSEP;
        optimal_result.RMSECV = RMSECV;
        optimal_result.CARS = CARS;
        optimal_result.ypred2_train = Ypred_train;
        optimal_result.Rank2 = select;
        optimal_result.Xtrain2 = Xtrain_CV;
        optimal_result.Ytrain2 = Ytrain_CV;
        optimal_result.Xtest2 = Xtest_CV;
        optimal_result.ytest2 = Ytest_CV;
        optimal_result.RPD = RPD;
    end
end

results_filename = sprintf('%s\\Results_R2_P=%.4f_Train.mat', result_folder, optimal_result.R2_P);
save(results_filename, 'optimal_result');

fig = figure(1);
hold on;
plot(optimal_result.ytest2, optimal_result.ypred2, '.', 'MarkerSize', 15);
plot(optimal_result.Ytrain2, optimal_result.ypred2_train, '.', 'MarkerSize', 15);
plot(min(optimal_result.ytest2):max(optimal_result.ytest2), min(optimal_result.ytest2):max(optimal_result.ytest2), 'r-', 'LineWidth', 1.5);
hold off;
xlabel('Actual Values'); ylabel('Predicted Values');
title(sprintf('R2C=%.3f,R2P=%.3f,RMSEC=%.3f,RMSEP=%.3f', optimal_result.R2_C, optimal_result.R2_P, optimal_result.RMSEC, optimal_result.RMSEP));
legend('Test Data', 'Train Data', 'Ideal Line', 'Location', 'NorthWest');
saveas(fig, sprintf('%s\\Results_R2_P=%.4f_Train.tif', result_folder, optimal_result.R2_P), 'tiff');
close(fig);
end
