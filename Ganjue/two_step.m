%% 1. 清理工作空间
clc;            % 清空命令窗口
clear;          % 清除所有变量
close all;      % 关闭所有图形窗口


%% 初始化
folder = 'D:\\Desktop';   % 本文件夹存储位置
physical_file_name = 'ganguan.xlsx';  % 理化值文件名
Smooth_folder = sprintf('%s\\Data processing\\Result\\Smooth\\ganjue\\Smooth_Results.mat',folder);% 光谱波形数据文件
Psyhical_data_folder = sprintf('%s\\Data processing\\data\\physical\\%s',folder,physical_file_name); % 样品理化值数据文件
load(Smooth_folder);  % 加载预处理后数据
Psyhical_data = xlsread(Psyhical_data_folder); % 理化值数据读取
Psyhical_data = Psyhical_data(:,1:2); % 理化值数据读取
data =  Post_smooth_data(1:120,:); %取前120个样本


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

[coeff, score, latent, ~, explained, ~] = pca(Xtrain);
Xtrain =  Xtrain  * coeff(:, 1:76);
% 使用相同的主成分变换应用于测试集
Xtest = Xtest  * coeff(:, 1:76);

%% 3. 定义网络结构
% 使用层级 API 定义多层感知机网络，各层说明如下：
inputLength = 50;  % 根据你的实际样本长度修改

layers = [
    sequenceInputLayer(inputLength)                        % 输入层
    convolution1dLayer(3, 16, 'Padding', 'same')           % 卷积层1（3核，16通道）
    batchNormalizationLayer
    reluLayer
    % maxPooling1dLayer(2, 'Stride', 2)

    convolution1dLayer(3, 32, 'Padding', 'same')           % 卷积层2（3核，32通道）
    batchNormalizationLayer
    reluLayer
    % maxPooling1dLayer(2, 'Stride', 2)

    convolution1dLayer(3, 64, 'Padding', 'same')           % 卷积层3（3核，64通道）
    batchNormalizationLayer
    reluLayer
    globalAveragePooling1dLayer                           % 全局平均池化（减少过拟合）

    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(10)                               % 输出10类
    softmaxLayer
    classificationLayer
];

%% 4. 配置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 4, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% 5. 训练网络
% 调用 trainNetwork 函数输入训练数据、网络层和训练选项进行模型训练
net = trainNetwork(Xtrain, Ytrain, layers, options);

%% 6. 使用训练好的模型进行预测
% 使用 predict 函数对新数据进行预测
YPred = classify(net, Xtest);
accuracy = mean(YPred == Ytest);
confusionchart(Ytest, YPred);  % 可视化混淆矩阵

YPred_train = classify(net, Xtrain);
accuracy_train = mean(YPred_train == Ytrain);
confusionchart(Ytrain, YPred_train);  % 可视化混淆矩阵





