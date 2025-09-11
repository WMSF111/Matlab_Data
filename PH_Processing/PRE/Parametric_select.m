clc;
clear;
close;

% 数据预处理
%% 初始化
folder = 'D:\\Desktop';   %本文件夹存储位置
physical_file_name = 'sweet.xlsx';  % 理化值文件名
% 光谱波形数据文件
Wave_folder = sprintf('%s\\Data processing\\data\\wavelength144.mat',folder);
% 采集样品光谱数据文件
Black_White_Data_folder = sprintf('%s\\Data processing\\Result\\Black_White\\post_processing_data.xlsx',folder);
Psyhical_data_folder = sprintf('%s\\Data processing\\data\\physical\\%s',folder,physical_file_name); % 样品理化值数据文件
% 平滑数据保存文件
smooth_mat_file_name = sprintf('%s\\Data processing\\Result\\Smooth\\Smooth_Results.mat', ...
                       folder);
smooth_file_name = sprintf('%s\\Data processing\\Result\\Smooth\\Smooth_Results.tif', ...
                   folder);  
%% 加载数据
data = xlsread(Black_White_Data_folder);  % 光谱数据读取
data =  data(1:120,:);
Psyhical_data = xlsread(Psyhical_data_folder); % 理化值数据读取
Good_RMSE = inf;
for i=5:2:21
    %% 数据预处理 SG+SNV+归一化
    SG_data = SG(data,2,i); %SG滤波

    SNV_data = SNV(SG_data); %SNV

    Zero_data = normalization(SNV_data,1); %数据归一化

    Post_smooth_data = Zero_data;    %用于下方数据更改方便

    [select,not_select] = KS(data,84); 

    %训练集
    Xtrain = Post_smooth_data(select,:);
    Ytrain = Psyhical_data(select,:);
    %测试集
    Xtest = Post_smooth_data(not_select,:);
    Ytest = Psyhical_data(not_select,:);

    CV = plscv(Xtrain, Ytrain, 256, 10, 'center');
    A2 = CV.optLV;

    % 使用选择的特征进行PLS建模
    PLS = pls(Xtrain, Ytrain, A2,  'center');
    Xtest_expand = [Xtest ones(size(Ytest, 1), 1)];
    coef = PLS.regcoef_original_all;
    %预测值
    Ypred = Xtest_expand * coef(:, end);
    Ypred_train = plsval(PLS,Xtrain,Ytrain);

    % 性能评估
    SST = sum((Ytest - mean(Ytest)).^2);
    SSE = sum((Ytest - Ypred).^2);
    R2_C = PLS.R2;
    R2_P = 1 - SSE / SST;
    RMSEC = sqrt(PLS.SSE / size(Xtrain, 1));
    RMSEP = sqrt(SSE / size(Xtest, 1));
    if RMSEP-RMSEC<Good_RMSE
        Good_RMSE = RMSEP-RMSEC
        Good_RMSEP = RMSEP;
        Good_RMSEC = RMSEC;
        Good_I = i;
    end   
end
fprintf('%f\r\n',Good_RMSE);
fprintf('%f\r\n',Good_RMSEP);
fprintf('%f\r\n',Good_RMSEC);
fprintf('%d\r\n',Good_I);


