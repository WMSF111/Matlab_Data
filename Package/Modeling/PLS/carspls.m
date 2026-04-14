function CARS=carspls(X,y,A,fold,method,num,selectLV,originalVersion,order) 
%+++ CARS: Competitive Adaptive Reweighted Sampling method for variable selection.
%+++ X: The data matrix of size m x p
%+++ y: The reponse vector of size m x 1
%+++ A: the maximal principle to extract.
%+++ fold: the group number for cross validation.
%+++ method: pretreatment method.
%+++ num: the  number of Monte Carlo Sampling runs.
%+++ CARS：用于变量选择的竞争性自适应重加权采样法�?
%+++ X：大小为 m x p 的数据矩�?
%+++ y:大小�?m x 1 的响应向�?
%+++ A：要提取的最大原则�?
%+++ fold：交叉验证的组数�?
%+++ method：预处理方法�?
%+++ num：蒙特卡罗采样运行次�?
%+++ selectLV:     0: selecting the optimal LV achieving the global minimum of RMSECV curves
%                  1: selecting the optimal LV achieving the maximum RMSECV within the range of global minimum + 1 standard deviation.
%+++ originalVersion:  1: using the original version of CARS which
%                         by design may give results which can not be
%                         exactly reproduced due to the embeding random
%                         sampling technique but the results can be
%                         statistically reproduced if you run CARS multiple
%                         times, say 50 times.This is default.
%                      0: using a simplified version of CARS with the random
%                         sampling element removed. So each run of CARS
%                         will give exactly the same results, which is also an                       a
%                         appreciated fature though "containing randomness"
%                         is the nature of data. Considering the 'exact'
%                         reproducibility, this simplified version is used
%                         as default. You can set this value to 1 if want
%                         to use the original version
%+++ order:  0: samples are sorted based on y values
%            1: samples are randomly ordered
%            2: samples are in the same order as in the orginal input
%+++ Hongdong Li, Dec.15, 2008.
%+++ Advisor: Yizeng Liang, yizeng_liang@263.net
%+++ lhdcsu@gmail.com
%+++ Ref:  Hongdong Li, Yizeng Liang, Qingsong Xu, Dongsheng Cao, Key
%    wavelengths screening using competitive adaptive reweighted sampling 
%    method for multivariate calibration, Anal Chim Acta 2009, 648(1):77-84

%+++ selectLV: 0: 选择达到 RMSECV 曲线全局最小值的最�?LV�?
% 1：在全局最小�?+ 1 个标准偏差的范围内选择 RMSECV 最大的最�?LV�?
%+++ originalVersion: 1: 使用原始版本�?CARS。该版本的设计可能导致无法由于采用了嵌入式随机但如果�?CARS 进行统计，结果可以但如果运行 CARS 多但如果多次运行 CARS，例�?50 次，则可以在统计上重�?% 的结果�?
% 0：使用简化版�?CARS，去掉了随机抽样元素�?
% 取样元素。因此每次运�?CARS都会得到完全相同的结果。虽�?"包含随机�?"是数据的本质，但也是一种值得赞赏的特性�?
%是数据的本质。考虑�?"精确考虑�?"精确 "的可重复性，这个简化版本被作为默认版本使用�?
% 作为默认值。如果您想使用原始版�?
%+++ 排序�? 0：样本根�?y 值排�?
% 1：样本随机排�?
% 2：样本顺序与原始输入相同
%+++ 李红东，2008 �?12 �?15 日�?
%+++ 导师：梁一�? yizeng_liang@263.net
%+++ lhdcsu@gmail.com
%+++ 参考文�? 李红�? 梁一�? 徐青�? 曹东�? 关键利用竞争性自适应加权采样筛选波长用于多变量校准�?% 方法，Anal Chim Acta�?009�?48�?）：77-84
tic;
%+++ Initial settings.
if nargin<9;order=0;end;
if nargin<8;originalVersion=0;end;
if nargin<7;selectLV=1;end;
if nargin<6;num=50;end;
if nargin<5;method='center';end;
if nargin<4;fold=5;end;
if nargin<3;A=2;end;

%+++ Initial settings.
[Mx,Nx]=size(X);
A=min([Mx Nx A]);
index=1:Nx;
ratio=0.9;
r0=1;
r1=2/Nx;
Vsel=1:Nx;
Q=floor(Mx*ratio);
W=zeros(Nx,num);
Ratio=zeros(1,num);

%+++ Parameter of exponentially decreasing function. 
b=log(r0/r1)/(num-1);  a=r0*exp(b);

%+++ Main Loop
for iter=1:num
     
     if originalVersion == 1
     perm=randperm(Mx);   
     Xcal=X(perm(1:Q),:); ycal=y(perm(1:Q));   %+++ Monte-Carlo Sampling.
     else
     Xcal=X;ycal=y;    
     end
     PLS=pls(Xcal(:,Vsel),ycal,A,method);    %+++ PLS model
     w=zeros(Nx,1);coef=PLS.regcoef_original(1:end-1,end);
     w(Vsel)=coef;W(:,iter)=w; 
     w=abs(w);                                  %+++ weights
     [ws,indexw]=sort(-w);                      %+++ sort weights
     
     ratio=a*exp(-b*(iter+1));                      %+++ Ratio of retained variables.
     Ratio(iter)=ratio;
     K=round(Nx*ratio);  
     
     w(indexw(K+1:end))=0;                          %+++ Eliminate some variables with small coefficients.  
     if originalVersion == 1
     if sum(w) > 0
     Vsel=unique(randsample(Nx,Nx,true,w));                 %+++ Reweighted Sampling from the pool of retained variables.
     else
     nz=find(w~=0);
     if isempty(nz)
     Vsel=unique(indexw(1:max(1,min(K,Nx))));
     else
     Vsel=unique(nz);
     end
     end
     else
     Vsel=unique(find(w~=0));
     if isempty(Vsel)
     Vsel=unique(indexw(1:max(1,min(K,Nx))));
     end
     end
%      fprintf('The %dth variable sampling finished.\n',iter);    %+++ Screen output.
 end

%+++  Cross-Validation to choose an optimal subset;
RMSECV=zeros(1,num);
Q2_max=zeros(1,num);
LV=zeros(1,num);
for i=1:num
   vsel=find(W(:,i)~=0);
 
   CV=plscv(X(:,vsel),y,A,fold,method,0,order);  
   if selectLV == 0
   RMSECV(i)=CV.RMSECV_min;
   Q2_max(i)=CV.Q2_max;   
   LV(i)=CV.optLV;
   elseif selectLV==1
   RMSECV(i)=CV.RMSECV_min_1SD;
   Q2_max(i)=CV.Q2_max_1SD;   
   LV(i)=CV.optLV_1SD; 
   end
   
%    fprintf('The %d/%dth subset finished.\n',i,num);
end
[RMSECV_min,indexOPT]=min(RMSECV);
Q2_max=max(Q2_max);




%+++ save results;
time=toc;
%+++ output
CARS.W=W;
CARS.time=time;
CARS.RMSECV=RMSECV;
CARS.RMSECV_min=RMSECV_min;
CARS.Q2_max=Q2_max;
CARS.iterOPT=indexOPT;
CARS.optLV=LV(indexOPT);
CARS.ratio=Ratio;
CARS.vsel=find(W(:,indexOPT)~=0)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





