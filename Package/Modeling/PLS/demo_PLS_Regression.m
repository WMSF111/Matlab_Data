%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  This script is used to test whether the functions in this  %%%%%%%%%%  
%%%%%%%%%%  package can run smoothly.                                  %%%%%%%%%%
%%%%%%%%%%  Suggest: every time you make some modification of codes    %%%%%%%%%%
%%%%%%%%%%  this pakcage, please run this script to debug. Notice that %%%%%%%%%%
%%%%%%%%%%  this script is only for testing so model parameters may not %%%%%%%%%
%%%%%%%%%%  be optimal.                                                %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  H.D. Li, lhdcsu@gmail.com                                  %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ Cross validation
load corn_m51;
A=6;
K=5;
method='center';
N=500;
Nmcs=50;
CV=plscv(X,y,A,K,method)
MCCV=plsmccv(X,y,A,method,N)
DCV=plsdcv(X,y,A,K,method,Nmcs)
RDCV=plsrdcv(X,y,A,K,method,Nmcs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ build a model and make predictions on test set.
load corn_m51;
Rank=ks(X); %+++ Data partition using Kennard-Stone algorithm
Xcal=X(Rank(1:60),:);
ycal=y(Rank(1:60),:);
Xtest=X(Rank(61:80),:);
ytest=y(Rank(61:80),:);
PLS=pls(Xcal,ycal,10);  %+++ Build a PLS regression model using training set
[ypred,RMSEP]=plsval(PLS,Xtest,ytest); %+++ make predictions on test set
figure;
plot(ytest,ypred,'.',ytest,ytest,'r-');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ Outlier detection 异常值检测 %N 蒙特卡罗采样次数，默认值：2500。
% ratio：随机抽取的样本比例构建PLS模型，默认为0.75
load corn_m51;
F=mcs(X,y,12,'center',1000,0.7)                                 
figure;
plotmcs(F);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ CARS-PLS for variable selection  CARS-特征波长筛选
load corn_m51;
A=10;
K=5; 
N=50;
%+++ X：大小为 m x p 的数据矩阵
%+++ y:大小为 m x 1 的响应向量
%+++ A：要提取的最大原则。
%+++ fold：交叉验证的组数。
%+++ method：预处理方法。
%+++ num：蒙特卡罗采样运行次数
CARS=carspls(X,y,A,K,method,N,0,1);  % original version of CARS 原版 CARS
figure;
plotcars(CARS);
CARS1=carspls(X,y,A,K,method,N,1,1);  % original version of CARS but with optimal LV selection using the Standard deviation information.
                                      %原版 CARS,但与最佳LV选择使用标准偏差信息.
figure;
plotcars(CARS1);
CARS2=carspls(X,y,A,K,method,N,0,0);  % exactly reproducible version of CARS with random elements removed
                                      % 删除随机样本
figure;
plotcars(CARS2);
CARS3=carspls(X,y,A,K,method,N,1,0);  % exactly reproducible version of CARS but with optimal LV selection using the Standard deviation information.
                                      %完全可重复的 CARS 版本，但使用标准偏差信息优化 LV 选择
figure;
plotcars(CARS3);


%+++ moving window PLS
[WP,RMSEF]=mwpls(X,y);
figure;
plot(WP,RMSEF);
xlabel('wavelength');
ylabel('RMSEF');

%+++ Random frog: here N is set to 1000 Only for testing. 随机青蛙：此处 N 设置为 1000 只用于测试。
%    N usually needs to be large, e.g. 10000 considering the huge variable N通常需要很大，例如考虑到巨大的变量，10000
%    space.
Frog=randomfrog_pls(X,y,A,method,1000,5);
figure;
plot(Frog.probability);
xlabel('variable index');
ylabel('selection probability');




