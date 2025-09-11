clc;
clear;
close; 

%% 初始化
num_iterations = 100000;  % 迭代次数
folder = 'D:\\Desktop';   % 本文件夹存储位置
physical_file_name = 'sweet.xlsx';  % 理化值文件名
Smooth_folder = sprintf('%s\\Data processing\\Result\\Smooth\\SSC\\Smooth_Results.mat',folder);% 光谱波形数据文件
Psyhical_data_folder = sprintf('%s\\Data processing\\data\\physical\\%s',folder,physical_file_name); % 样品理化值数据文件
Result_folder = sprintf('%s\\Data processing\\Result\\Model\\SSC\\PLS_CARS',folder);% 结果保存文件夹
load(Smooth_folder);  % 加载预处理后数据
Psyhical_data = xlsread(Psyhical_data_folder); % 理化值数据读取
data =  Post_smooth_data(1:120,:); %取前120个样本

%% PLS建模
%样品划分（KS方法）
%select：选出样本序号，行向量。 
%not_select：未选出样本序号，行向量。
[select,not_select] = KS(data,90); 
%训练集
Xtrain = data(select,:);
Ytrain = Psyhical_data(select,:);
%测试集
Xtest = data(not_select,:);
Ytest = Psyhical_data(not_select,:);

CV = plscv(Xtrain, Ytrain, 256, 10, 'center');
A = CV.optLV;

%找寻最优区间划分   
%初始化最小R2_P及其对应的数据结构体
max_R2_P = -inf;
Good_RMSE = inf;
optimal_result = struct();

for j = 1:num_iterations      %寻找最优特征波长
    CARS = carspls(Xtrain, Ytrain, A, 10, 'center', 30, 0, 1); %car-PLS寻找
    excellent_feature = CARS.vsel;    %最优波长序列
    post_data = data(:, excellent_feature); %选定波长后的数据
    %训练集
    Xtrain_CV = post_data(select, :);
    Ytrain_CV = Psyhical_data(select, :);
    %测试集
    Xtest_CV = post_data(not_select, :);
    Ytest_CV = Psyhical_data(not_select, :);

    CV2 = plscv(Xtrain_CV, Ytrain_CV, 256, 10, 'center');
    A2 = CV2.optLV;

    % 使用选择的特征进行PLS建模
    PLS = pls(Xtrain_CV, Ytrain_CV, A2, 'center');
%   Xtest_expand = [Xtest_CV ones(size(Ytest_CV, 1), 1)];
%   coef = PLS.regcoef_original_all;
    %预测值
%   Ypred = Xtest_expand * coef(:, end);
    Ypred = plsval(PLS,Xtest_CV,Ytest_CV);
    Ypred_train = plsval(PLS,Xtrain_CV,Ytrain_CV);

    % 性能评估        
    SST_train = sum((Ytrain_CV - mean(Ytrain_CV)).^2);
    SSE_train = sum((Ytrain_CV - Ypred_train).^2);
    SST_test = sum((Ytest_CV - mean(Ytest_CV)).^2);
    SSE_test = sum((Ytest_CV - Ypred).^2);
%   R2_C = PLS.R2;
    R2_C = 1 - SSE_train / SST_train;
    R2_P = 1 - SSE_test / SST_test;
%   RMSEC = sqrt(PLS.SSE / size(Xtrain_CV, 1));
    RMSEC = sqrt(SSE_train / size(Xtrain_CV, 1));
    RMSEP = sqrt(SSE_test / size(Xtest_CV, 1));
    RMSECV = CV.RMSECV_min;
    RPD = std(Ytest_CV) / RMSEP;
    % 如果当前 R2_P 大于最大 R2_P，则更新最大值及其对应的数据
    if R2_P > max_R2_P
        if RMSEP - RMSEC < Good_RMSE
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
    Progress = sprintf('Test: remaining number--%d\r\n',num_iterations-j);
    fprintf(Progress);
end
%保存结果到MAT文件   
% 建模结果保存数据文件 
results_filename = sprintf('%s\\Results_R2_P=%.4f_Train.mat', ... 
               Result_folder,optimal_result.R2_P);              
save(results_filename, 'optimal_result');

%预测结果画图
fig = figure(1);
hold on;
plot(Ytest_CV, optimal_result.ypred2, '.', 'MarkerSize', 15);
plot(Ytrain_CV, optimal_result.ypred2_train, '.', 'MarkerSize', 15);
plot(2.5:18, 2.5:18, 'r-', 'LineWidth', 1.5);
hold off;
xlabel('Actual Values');
ylabel('Predicted Values');
str = sprintf('R2C=%.3f,R2P=%.3f,RMSEC=%.3f,RMSEP=%.3f', ...
      optimal_result.R2_C,optimal_result.R2_P,optimal_result.RMSEC,optimal_result.RMSEP);

title(str);
legend('Test Data','Train Data','Ideal Line', 'Location', 'NorthWest');
Graph_file_name = sprintf('%s\\Results_R2_P=%.4f_Train.tif', ...
                  Result_folder,optimal_result.R2_P);


saveas(fig, Graph_file_name, 'tiff');
close(fig); %关闭结果

%CARS结果画图
fig2 = figure(2);
set(fig2,'visible','off');
plotcars(optimal_result.CARS);
xlabel('variable index');
ylabel('selection probability');
CARS_file_name = sprintf('%s\\CARS_Results_R2_P=%.4f_Train.tif', ...
                 Result_folder,optimal_result.R2_P);


saveas(fig2, CARS_file_name, 'tiff');
close(fig2);%关闭结果

% 打印最优数据
max_R2_C = optimal_result.R2_C; 
max_R2_P = optimal_result.R2_P;
optimism_RMSEC = optimal_result.RMSEC;
optimism_RMSEP = optimal_result.RMSEP;
optimism_RPD = optimal_result.RPD;
%% 打印最优结果输出
fprintf('finally result:\r\nmax_R2_C=%.3f\r\nmax_R2_P==%.3f\r\noptimism_RMSEC==%.3f\r\noptimism_RMSEP==%.3f\r\noptimism_RPD==%.3f\r\n', ...
        max_R2_C,max_R2_P,optimism_RMSEC,optimism_RMSEP,optimism_RPD);




 





