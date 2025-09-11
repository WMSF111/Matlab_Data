clc;
clear;
close; 

%% 初始化
folder = 'D:\\Desktop';   % 本文件夹存储位置
physical_file_name = 'PH.xlsx';  % 理化值文件名
Smooth_folder = sprintf('%s\\Data processing\\Result\\Smooth\\PH\\Smooth_Results.mat',folder);% 光谱波形数据文件
Psyhical_data_folder = sprintf('%s\\Data processing\\data\\physical\\%s',folder,physical_file_name); % 样品理化值数据文件
Result_folder = sprintf('%s\\Data processing\\Result\\Model\\PH\\PLS_SPA',folder);% 结果保存文件夹
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


max_R2_P = -inf;
for s = 1 : size(Xtrain,2)
    for k = 30 : 90
        %% SPA执行
        wave_select = SPA(Xtrain,s,k);
        Xtrain_SPA = Xtrain(:,wave_select);  
        Xtest_SPA = Xtest(:,wave_select);  
        CV = plscv(Xtrain_SPA, Ytrain, 256, 10, 'center');
        A = CV.optLV;
        % 使用选择的特征进行PLS建模
        PLS = pls(Xtrain_SPA, Ytrain, A, 'center');
        Ypred = plsval(PLS,Xtest_SPA,Ytest);
        Ypred_train = plsval(PLS,Xtrain_SPA,Ytrain);
        % 性能评估        
        SST_train = sum((Ytrain - mean(Ytrain)).^2);
        SSE_train = sum((Ytrain - Ypred_train).^2);
        SST_test = sum((Ytest - mean(Ytest)).^2);
        SSE_test = sum((Ytest - Ypred).^2);
        R2_C = 1 - SSE_train / SST_train;
        R2_P = 1 - SSE_test / SST_test;
        RMSEC = sqrt(SSE_train / size(Xtrain_SPA, 1));
        RMSEP = sqrt(SSE_test / size(Xtest_SPA, 1));
        RMSECV = CV.RMSECV_min;
        RPD = std(Ytest) / RMSEP;
        if R2_P > max_R2_P
            max_R2_P = R2_P; 
            optimal_result.A = A;
            optimal_result.SPAK = k;
            optimal_result.SPAS = s;
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
        end
    end
     fprintf("剩余次数%d\r\n",size(Xtrain,2)-s);
end
%% 保存结果到MAT文件   
% 建模结果保存数据文件 
results_filename = sprintf('%s\\Results_R2_P=%.4f_Train.mat', ... 
               Result_folder,optimal_result.R2_P);              
save(results_filename, 'optimal_result');

%% 预测结果画图
fig = figure(1);
hold on;
plot(Ytest, optimal_result.ypred2, '.', 'MarkerSize', 15);
plot(Ytrain, optimal_result.ypred2_train, '.', 'MarkerSize', 15);
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

%% 打印最优数据
max_R2_C = optimal_result.R2_C; 
max_R2_P = optimal_result.R2_P;
optimism_RMSEC = optimal_result.RMSEC;
optimism_RMSEP = optimal_result.RMSEP;
optimism_RPD = optimal_result.RPD;
%% 打印最优结果输出
fprintf('finally result:\r\nmax_R2_C=%.3f\r\nmax_R2_P==%.3f\r\noptimism_RMSEC==%.3f\r\noptimism_RMSEP==%.3f\r\noptimism_RPD==%.3f\r\noptimism_k==%d\r\noptimism_s==%d\r\n', ...
        max_R2_C,max_R2_P,optimism_RMSEC,optimism_RMSEP,optimism_RPD,optimal_result.SPAK,optimal_result.SPAS);
    