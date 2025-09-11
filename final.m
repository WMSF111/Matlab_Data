clc;
clear;
close;

folder = 'D:\\Desktop';   %本文件夹存储位置

%% 原始光谱数据绘制
Wave_folder = sprintf('%s\\Data processing\\data\\wavelength144.mat',folder);% 光谱波形数据文件
load(Wave_folder);                                                           % 波长数据读取

Black_White_Data_folder = sprintf('%s\\Data processing\\Result\\Black_White\\post_processing_data.xlsx',folder);
data = xlsread(Black_White_Data_folder);               % 光谱数据读取
data =  data(1:120,:);                                 % 120个样本

fig = figure(1);
plot(wavelength144,data,'LineWidth', 0.9);    % 原始光谱
title('Raw spectra','FontName','Times New Rome','FontSize',12,'FontWeight', 'bold'); % 图像标题
xlabel('Wavelength(nm)','FontName','Times New Rome','FontSize',11);   %x轴标题
ylabel('Reflectance','FontName','Times New Rome','FontSize',11);%y轴标题
axis([575,1100,0.1,0.8]); %设置范围


%% 相关性分析 Spearman 和 Pearson 
FF_folder = sprintf('%s\\Data processing\\data\\physical\\yingdu.xlsx',folder); 
SSC_folder = sprintf('%s\\Data processing\\data\\physical\\sweet.xlsx',folder); 
PH_folder = sprintf('%s\\Data processing\\data\\physical\\PH.xlsx',folder); 
ganguan_folder = sprintf('%s\\Data processing\\data\\physical\\ganguan.xlsx',folder);   % 样品理化值数据文件
FF_Value = xlsread(FF_folder);
SSC_Value = xlsread(SSC_folder);  
PH_Value = xlsread(PH_folder);
ganguan_Value = xlsread(ganguan_folder); 
ganguan_Value = ganguan_Value(:,6:10); % 理化值数据读取
data = [FF_Value, SSC_Value, PH_Value,ganguan_Value];

data = xlsread('D:\\Desktop\\yu.xlsx');
% 计算 Pearson 相关系数
[R, P] = corr(data, 'Type', 'Pearson');

% 显示结果
disp('Pearson 相关系数矩阵:');
disp(R);
disp('Pearson 相关性 p 值矩阵:');
disp(P);

%% 预处理结果画图
SG_data = SG(data,3,13); %SG滤波
MSC_data = MSC(SG_data);

fig2 = figure(2);
plot(wavelength144,SG_data,'LineWidth', 0.9);    %原始光谱
title("SG",'FontName','Times New Rome','FontSize',12,'FontWeight', 'bold');        %添加标题
xlabel('Wavelength(nm)','FontName','Times New Rome','FontSize',11);   %x轴标题
ylabel('Reflectance','FontName','Times New Rome','FontSize',11);%y轴标题
axis([575,1100,0.15,0.6]); %设置范围
fig3 = figure(3);
plot(wavelength144,MSC_data,'LineWidth', 0.9); %SG光谱
title("SG+MSC",'FontName','Times New Rome','FontSize',12,'FontWeight', 'bold');           %添加标题
xlabel('Wavelength(nm)','FontName','Times New Rome','FontSize',11);   %x轴标题
ylabel('Reflectance','FontName','Times New Rome','FontSize',11);%y轴标题
axis([575,1100,0.1,0.65]); %设置范围


%% 结果图绘制
% 硬度
fig4 = figure(4);
hold on;
set(gca, 'Color', [0.98, 0.98, 0.98]); % 修改背景色
plot(-10:90, -10:90, '-', 'Color', [31, 119, 180] / 255, 'LineWidth', 2,'HandleVisibility', 'off'); % 直线

% 标记点
scatter(optimal_result.Ytrain2, optimal_result.ypred2_train,40, [0, 127, 95] / 255, 'o', 'MarkerFaceAlpha', 0.5,'LineWidth', 1.5);
scatter(optimal_result.ytest2, optimal_result.ypred2,40,[230, 57, 70]/ 255, '+', 'LineWidth', 1.5);

% 调整字体
title('RFE-PLS','FontName','Times New Rome','FontSize',15,'FontWeight','bold');
xlabel('Actual Value(N)','FontName','Times New Rome','FontSize',13);
ylabel('Predicted Value(N)','FontName','Times New Rome','FontSize',13);

Tag1 = sprintf('$R_C^2 = %.3f \\  RMSEC = %.3f$',optimal_result.R2_C,optimal_result.RMSEC);
Tag2 = sprintf('$R_P^2 = %.3f \\  RMSEP = %.3f$',optimal_result.R2_P,optimal_result.RMSEP);
% 绘制图例
h = legend({Tag1,Tag2}, 'Location', 'NorthWest','Interpreter', 'latex');
set(h, 'FontName', 'Times New Rome', 'FontSize', 12);
% 关闭图例边框
legend boxoff;

hold off;

% SSC
fig5 = figure(5);
hold on;
set(gca, 'Color', [0.98, 0.98, 0.98]); % 修改背景色
plot(2:18, 2:18, '-', 'Color', [31, 119, 180] / 255, 'LineWidth', 2,'HandleVisibility', 'off'); % 直线

% 标记点
scatter(optimal_result.Ytrain2, optimal_result.ypred2_train,40, [0, 127, 95] / 255, 'o', 'MarkerFaceAlpha', 0.5,'LineWidth', 1.5);
scatter(optimal_result.ytest2, optimal_result.ypred2,40,[230, 57, 70]/ 255, '+', 'LineWidth', 1.5);

% 调整字体
title('RFE-PLS','FontName','Times New Rome','FontSize',15,'FontWeight','bold');
xlabel('Actual Value(°Brix)','FontName','Times New Rome','FontSize',13);
ylabel('Predicted Value(°Brix)','FontName','Times New Rome','FontSize',13);

Tag1 = sprintf('$R_C^2 = %.3f \\  RMSEC = %.3f$',optimal_result.R2_C,optimal_result.RMSEC);
Tag2 = sprintf('$R_P^2 = %.3f \\  RMSEP = %.3f$',optimal_result.R2_P,optimal_result.RMSEP);
% 绘制图例
h = legend({Tag1,Tag2}, 'Location', 'NorthWest','Interpreter', 'latex');
set(h, 'FontName', 'Times New Rome', 'FontSize', 12);
% 关闭图例边框
legend boxoff;

hold off;

% PH值
fig6 = figure(6);
hold on;
set(gca, 'Color', [0.98, 0.98, 0.98]); % 修改背景色
plot(2:5, 2:5, '-', 'Color', [31, 119, 180] / 255, 'LineWidth', 2,'HandleVisibility', 'off'); % 直线

% 标记点
scatter(optimal_result.Ytrain2, optimal_result.ypred2_train,40, [0, 127, 95] / 255, 'o', 'MarkerFaceAlpha', 0.5,'LineWidth', 1.5);
scatter(optimal_result.ytest2, optimal_result.ypred2,40,[230, 57, 70]/ 255, '+', 'LineWidth', 1.5);

% 调整字体
title('RFE-PLS','FontName','Times New Rome','FontSize',15,'FontWeight','bold');
xlabel('Actual Value','FontName','Times New Rome','FontSize',13);
ylabel('Predicted Value','FontName','Times New Rome','FontSize',13);

Tag1 = sprintf('$R_C^2 = %.3f \\  RMSEC = %.3f$',optimal_result.R2_C,optimal_result.RMSEC);
Tag2 = sprintf('$R_P^2 = %.3f \\  RMSEP = %.3f$',optimal_result.R2_P,optimal_result.RMSEP);
% 绘制图例
h = legend({Tag1,Tag2}, 'Location', 'NorthWest','Interpreter', 'latex');
set(h, 'FontName', 'Times New Rome', 'FontSize', 12);
% 关闭图例边框
legend boxoff;

hold off;

%% 波长选择图
% CARS
folder = 'D:\\Desktop';   %本文件夹存储位置
Wave_folder = sprintf('%s\\Data processing\\data\\wavelength144.mat',folder);% 光谱波形数据文件
load(Wave_folder);                                                           % 波长数据读取
Black_White_Data_folder = sprintf('%s\\Data processing\\Result\\Black_White\\post_processing_data.xlsx',folder);
data = xlsread(Black_White_Data_folder);               % 光谱数据读取
data =  data(1,:); 
data = move_smooth(data,11);
fig7 = figure(7);
set(gca, 'Color', [0.98, 0.98, 0.98]); % 修改背景色
hold on;
plot(wavelength144,data,'LineWidth', 2,'Color',[0,51,102]/256);    % 原始光谱

scatter(wavelength144(:,optimal_result.selected_features),data(:,optimal_result.selected_features),40, [128,0,32] / 255, 'o','LineWidth',2);

title('CARS','FontName','Times New Rome','FontSize',15,'FontWeight','bold');
xlabel('Wavelength(nm)','FontName','Times New Rome','FontSize',13);
ylabel('Reflectance','FontName','Times New Rome','FontSize',13);
axis([575,1100,0.1,0.7]); %设置范围
hold off;



% RFE
folder = 'D:\\Desktop';   %本文件夹存储位置
Wave_folder = sprintf('%s\\Data processing\\data\\wavelength144.mat',folder);% 光谱波形数据文件
load(Wave_folder);                                                           % 波长数据读取
Black_White_Data_folder = sprintf('%s\\Data processing\\Result\\Black_White\\post_processing_data.xlsx',folder);
data = xlsread(Black_White_Data_folder);               % 光谱数据读取
data =  data(1,:); 
data = move_smooth(data,11);
fig8 = figure(8);
set(gca, 'Color', [0.98, 0.98, 0.98]); % 修改背景色
hold on;
plot(wavelength144,data,'LineWidth', 2,'Color',[0,51,102]/256);    % 原始光谱

scatter(wavelength144(:,optimal_result.wave_select),data(:,optimal_result.wave_select),40, [128,0,32] / 255, 'o', 'MarkerFaceAlpha', 0.7,'LineWidth',1);

title('RFE','FontName','Times New Rome','FontSize',15,'FontWeight','bold');
xlabel('Wavelength(nm)','FontName','Times New Rome','FontSize',13);
ylabel('Reflectance','FontName','Times New Rome','FontSize',13);
axis([575,1100,0.1,0.7]); %设置范围
hold off;


% SPA
folder = 'D:\\Desktop';   %本文件夹存储位置
Wave_folder = sprintf('%s\\Data processing\\data\\wavelength144.mat',folder);% 光谱波形数据文件
load(Wave_folder);                                                           % 波长数据读取
Black_White_Data_folder = sprintf('%s\\Data processing\\Result\\Black_White\\post_processing_data.xlsx',folder);
data = xlsread(Black_White_Data_folder);               % 光谱数据读取
data =  data(1,:); 
data = move_smooth(data,11);
fig9 = figure(9);
set(gca, 'Color', [0.98, 0.98, 0.98]); % 修改背景色
hold on;
plot(wavelength144,data,'LineWidth', 2,'Color',[0,51,102]/256);    % 原始光谱

scatter(wavelength144(:,optimal_result.wave_select),data(:,optimal_result.wave_select),40, [128,0,32] / 255, 'o','MarkerFaceAlpha', 0.7,'LineWidth',1);

title('SPA','FontName','Times New Rome','FontSize',15,'FontWeight','bold');
xlabel('Wavelength(nm)','FontName','Times New Rome','FontSize',13);
ylabel('Reflectance','FontName','Times New Rome','FontSize',13);
axis([575,1100,0.1,0.7]); %设置范围
hold off;












