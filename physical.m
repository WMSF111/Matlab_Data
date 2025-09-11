%% 1. 清理工作空间
clc;            % 清空命令窗口
clear;          % 清除所有变量
close all;      % 关闭所有图形窗口


%% 初始化
folder = 'D:\\Desktop';   % 本文件夹存储位置
physical_file_name = 'ganguan.xlsx';  % 理化值文件名
ssc_file_name = 'sweet.xlsx';  % 理化值文件名
ph_file_name = 'PH.xlsx';  % 理化值文件名
FF_file_name = 'yingdu.xlsx';  % 理化值文件名
ssc_data_folder = sprintf('%s\\Data processing\\data\\physical\\%s',folder,ssc_file_name); % 样品理化值数据文件
ph_data_folder = sprintf('%s\\Data processing\\data\\physical\\%s',folder,ph_file_name); % 样品理化值数据文件
FF_data_folder = sprintf('%s\\Data processing\\data\\physical\\%s',folder,FF_file_name); % 样品理化值数据文件
Psyhical_data_folder = sprintf('%s\\Data processing\\data\\physical\\%s',folder,physical_file_name); % 样品理化值数据文件
Psyhical_data = xlsread(Psyhical_data_folder); % 理化值数据读取
Psyhical_data = Psyhical_data(:,6:10); % 理化值数据读取
SSC_data = xlsread(ssc_data_folder);
SSC_data = SSC_data./max(SSC_data);
PH_data = xlsread(ph_data_folder);
PH_data = PH_data./max(PH_data);
FF_data = xlsread(FF_data_folder);
FF_data = FF_data./max(FF_data);
data =  PH_data;
data = data(1:120,:);



%% 样品划分（KS方法）
%select：选出样本序号，行向量。 
%not_select：未选出样本序号，行向量。
[select,not_select] = KS(data,90); 
%训练集
Xtrain = data(select,:);
Ytrain = Psyhical_data(select,:);
%测试集
Xtest = data(not_select,:);
Ytest = Psyhical_data(not_select,:);

%% 3. 定义网络结构
% 使用层级 API 定义多层感知机网络，各层说明如下：
layers = [
    featureInputLayer(1, 'Name', 'input')               % 输入层，定义输入特征的维度
    fullyConnectedLayer(64, 'Name', 'fc1')              % 第一全连接层，256个节点
    eluLayer('Name', 'elu1', 'Alpha', 1)                 % ReLU激活函数
 %  dropoutLayer(0.6, 'Name', 'dropout1')                % Dropout层，丢弃率20%
    fullyConnectedLayer(32, 'Name', 'fc2')              % 第二全连接层，64个节点
    eluLayer('Name', 'elu2', 'Alpha', 1)                 % ReLU激活函数
%     dropoutLayer(0.4, 'Name', 'dropout2')               % Dropout层，丢弃率20%
    fullyConnectedLayer(16, 'Name', 'fc3')               % 第三全连接层，32个节点
    eluLayer('Name', 'elu3', 'Alpha', 1)                 % ReLU激活函数
%    dropoutLayer(0.1, 'Name', 'dropout3')   
    fullyConnectedLayer(8, 'Name', 'fc4')               % 第4全连接层，32个节点
    eluLayer('Name', 'elu4', 'Alpha', 1)                 % ReLU激活函数
    fullyConnectedLayer(5, 'Name', 'fc_output')          % 输出层，节点数与目标变量数一致
    regressionLayer('Name', 'regression')                % 回归层，自动采用均方误差损失
    ];

% layers = [
%     featureInputLayer(30, 'Name', 'input')               % 输入层，定义输入特征的维度
%     fullyConnectedLayer(512, 'Name', 'fc1')              % 第一全连接层，256个节点
%     sigmoidLayer('Name', 'sigmoid1')                     % Sigmoid 激活函数
% %     eluLayer('Name', 'elu1', 'Alpha', 3)                 % ReLU激活函数
%     fullyConnectedLayer(128, 'Name', 'fc2')              % 第一全连接层，256个节点
%     sigmoidLayer('Name', 'sigmoid3')                % ReLU激活函数
% %     dropoutLayer(0.2, 'Name', 'dropout1')                % Dropout层，丢弃率20%
%     fullyConnectedLayer(32, 'Name', 'fc3')               % 第二全连接层，64个节点
% %     eluLayer('Name', 'elu3', 'Alpha', 3)                 % ReLU激活函数
% %     fullyConnectedLayer(32, 'Name', 'fc4')               % 第二全连接层，64个节点
%     sigmoidLayer('Name', 'sigmoid2')                     % Sigmoid 激活函数     
%     fullyConnectedLayer(5, 'Name', 'fc_output')          % 输出层，节点数与目标变量数一致
%     regressionLayer('Name', 'regression')                % 回归层，自动采用均方误差损失
%     ];
%% 4. 配置训练选项
% 使用 trainingOptions 设置优化器、训练轮次、批次大小及其它参数
options = trainingOptions('adam', ...          % 使用 Adam 优化器
    'MaxEpochs', 200, ...                         % 最大训练轮次设置为50
    'MiniBatchSize', 4, ...                     % 每个批次的样本数设为32
    'InitialLearnRate', 1e-3, ...                % 初始学习率1e-3
    'Shuffle', 'every-epoch', ...                % 每个epoch结束后打乱数据
    'ValidationFrequency', 10, ...               % 每训练30个mini-batch验证一次（如有验证集，可配置）
    'Verbose', true, ...                         % 显示训练中的信息
    'Plots', 'training-progress');               % 显示训练进度图

%% 5. 训练网络
% 调用 trainNetwork 函数输入训练数据、网络层和训练选项进行模型训练
net = trainNetwork(Xtrain, Ytrain, layers, options);

%% 6. 使用训练好的模型进行预测
% 使用 predict 函数对新数据进行预测
YPred = predict(net, Xtest);
YPred = round(min(max(YPred, 0), 10));
% YPred = round(YPred);

YPred_train = predict(net, Xtrain);
% YPred_train = round(min(max(YPred_train, 0), 10));
YPred_train = round(YPred_train);
% 显示预测结果
disp('预测结果：');
disp(YPred);

% 性能评估        
SST_train = sum((Ytrain - mean(Ytrain)).^2);
SSE_train = sum((Ytrain - YPred_train).^2);
SST_test = sum((Ytest - mean(Ytest)).^2);
SSE_test = sum((Ytest - YPred).^2);
R2_C = 1 - SSE_train / SST_train;
R2_P = 1 - SSE_test / SST_test;
RMSEC = sqrt(SSE_train / size(Xtrain, 1));
RMSEP = sqrt(SSE_test / size(Xtest, 1));
RPD = std(Ytest) / RMSEP;