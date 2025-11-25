%数据预处理、特征选择、模型建立以及性能评估
clc; %清理命令行
clear; %清理变量
close; %关闭所有弹窗

% 数据预处理
%% 初始化
folder = 'E:\study\algriothm\matlab';   %本文件夹存储位置
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
Good_RMSE = inf; %初始化为正无穷
for i=5:2:21
%% 数据预处理 SG+SNV+归一化
    SG_data = SG(data,2,i); %SG滤波

    SNV_data = SNV(SG_data); %SNV

    Zero_data = normalization(SNV_data,1); %数据归一化

    Post_smooth_data = Zero_data;    %用于下方数据更改方便

    [select,not_select] = KS(data,84);  % 区分训练和测试集
    
    %分别平滑和选择训练和测试集
    %训练集
    Xtrain = Post_smooth_data(select,:); % 从 Post_smooth_data 数据集中选择 select 中指定的行
    Ytrain = Psyhical_data(select,:);
    %测试集
    Xtest = Post_smooth_data(not_select,:);
    Ytest = Psyhical_data(not_select,:);
 %% 建模
    %此代码主要用于实现PLS模型的交叉验证，选择最佳的潜在变量数，并通过计算误差指标（如RMSECV、Q²等）来评估模型的性能。
    CV = plscv(Xtrain, Ytrain, 256, 10, 'center');
    A2 = CV.optLV;

    % 使用选择的特征进行PLS建模
    PLS = pls(Xtrain, Ytrain, A2,  'center');
    % 将原始的 Xtest 数据和一列全 1 的常数列连接起来，形成一个新的输入矩阵。
    % ones(size(Ytest, 1), 1) 创建一个列向量，长度为 Ytest 的行数（即测试样本数），其值全部为 1。
    Xtest_expand = [Xtest ones(size(Ytest, 1), 1)];
    %存储这个回归系数矩阵。（A2列）
    coef = PLS.regcoef_original_all;
    % 预测值
    % 扩展后的测试数据 Xtest_expand 与回归系数相乘，得到预测值 Ypred。
    Ypred = Xtest_expand * coef(:, end);
    % 计算模型在训练集上的预测结果。
    Ypred_train = plsval(PLS,Xtrain,Ytrain);

 %% 性能评估
     %.^2 是逐元素运算符（element-wise operator），表示对每个元素进行平方操作。
    %^2 是矩阵乘方运算符（matrix exponentiation），通常用于矩阵的乘方，而不是逐元素操作。
    SST = sum((Ytest - mean(Ytest)).^2); % 总平方和
    SSE = sum((Ytest - Ypred).^2); % 残差平方和
    R2_C = PLS.R2; %  训练集上的决定系数（R2）
    R2_P = 1 - SSE / SST; %测试集上的决定系数，越接近1表示模型的预测越好。
    RMSEC = sqrt(PLS.SSE / size(Xtrain, 1)); %训练集上的均方根误差
    RMSEP = sqrt(SSE / size(Xtest, 1)); % 测试集上的均方根误差
    if RMSEP-RMSEC<Good_RMSE % 测试集误差 RMSEP 和 训练集误差 RMSEC 之间的差距是否小于 Good_RMSE
        %如果差距小于 Good_RMSE，则说明模型在测试集上的性能较好，并且与训练集的误差差异不大。
        % Good_RMSE 是一个存储着当前模型性能的变量，通常用于保留最佳模型
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


