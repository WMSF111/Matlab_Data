function F=spa(X,y,A,K,Q,N,ratio,method,prior,PROCESS)
%+++ Subwindow Permutation Analysis for variable assessment.用于变量评估的排列分析。
%+++ Input:  X: m x p  (Sample matrix)
%            y: m x 1  (measured property)
%            A: The allowed maximum number of PLS components for cross-validation  用于交叉验证的 PLS 成分的最大允许数量
%            K: fold number for cross validation  交叉验证的折叠数
%            Q: The number of variables to be sampled in each MCS. 每个 MCS 中要采样的变量数。
%            N: The number of Monte Carlo Simulation. 蒙特卡罗模拟的数量。
%        ratio: The ratio of calibration samples to the total samples.  校准样本与总样本的比率。
%       method: pretreatment method. Contains:
%       autoscaling,pareto,minmax,center    预处理方法。包含：自动缩放，帕累托，最小最大值，中心
%        prior: prior probability of positive class. Default to 0 for  正分类的先验概率。根据数据计算先验概率时，
%        computing prior from data..say ldapinv.m f    or details    

% +++ Output: Structural data: F with items:
%         time: time cost in minute
%        model: a matrix of size N x p with element 0 or 1(means the variable is selected).
%          nLV: The optimal number of PLS components for each submodel.
%       error0: The normal prediction errors of the N submodels.
%       error1: The permutation prediction errors of the N submodels
%     interfer: a vector of size p. '1' indicates the variable is interferring
%            p: the p-value of each variable resulting from SPA. Note that
%               if a variable is interfering, the its p is manually added by 1, that is p=1+p.
%               This means that the variable of p>1 is an interferring one. It should be removed 
%               when building a classification model.
%         COSS: A metric for evaluating the importance of each variable,
%               COSS=-ln(p). According to the calculation of p,an
%               interferring variable will have a minus COSS score. This is
%               a trick.
%+++ Ranked variable: Ranked list of vairables.
%+++ Copy right (C) 2008 Hongdong Li
%+++ Hongdong Li, Oct. 16, 2008.
%+++ Revised in Nov.23, 2009.

if nargin<10;PROCESS=1;end;
if nargin<9;prior=0;end;
if nargin<8;method='autoscaling';end
if nargin<7;ratio=0.75;end
if nargin<6;N=1000;end
if nargin<5;Q=10;end;
if nargin<4;K=5;end
if nargin<3;A=2;end




[Mx,Nx]=size(X);

Qs=floor(Mx*ratio);
error0=zeros(N,1);
error1=nan(N,Nx);
interfer=zeros(1,Nx);
nLV=zeros(N,1);
ntest=Mx-Qs;
tic;
i=1;
for i=1:N
    ns=randperm(Mx); 
    calk=ns(1:Qs);
    
    testk=ns(Qs+1:end); % sampling in sample space
    nv=randperm(Nx); 
    nv=nv(1:Q);
    variableIndex=zeros(1,Nx);
    variableIndex(nv)=1;   
    
    Xcal=X(calk,nv);
    ycal=y(calk);
    
    Xtest=X(testk,nv);
    ytest=y(testk);    
       
    %data pretreatment    
    [Xcal,para1,para2]=pretreat(Xcal,method);
    Xtest=pretreat(Xtest,method,para1,para2);    
    clear para1 para2;
    
    %+++ Select the number of latent variables of PLS-LDA
    CV=plsldacv(Xcal,ycal,A,K,method,prior,0);
    nLV(i)=CV.optLV;
    %+++ Build a PLS-LDA model
    LDA=plslda(Xcal,ycal,CV.optLV,method,prior);
    %+++ Make predictions using the established PLS-LDA 
    [y_est,error_rate]=plsldaval(LDA,Xtest,ytest);
    %+++ Record prediction errors
    error0(i)=error_rate;
    error_temp=nan(1,Nx);
    
    %+++ Make predictions on the permutated sub-datset.
    for j=1:Q
      rn=randperm(ntest);
      Xtestr=Xtest;
      vi=Xtest(:,j);
      Xtestr(:,j)=vi(rn);
      [y_est,error_rate]=plsldaval(LDA,Xtestr,ytest);  
      error_temp(nv(j))=error_rate;     %+++ record permuted prediction errors
    end
    error1(i,:)=error_temp;
%     fprintf(fidw,'%s\n',num2str(error_temp));  %+++ Store the variables in the txt file.    
    if PROCESS==1; fprintf('The %d/%dth Monte Carlo sampling finished.\n',i,N);end   
end

%+++ p-value computing
p=zeros(1,Nx);
DMEAN=zeros(Nx,1);
DSD=zeros(Nx,1);
for i=1:Nx;
    k=find(~isnan(error1(:,i))==1);
    errori=error1(:,i);
    errorn=error0(k);
    errorp=errori(k);
  
    MEANn=mean(errorn);
    MEANp=mean(errorp);
    SDn=std(errorn);
    SDp=std(errorp);
    DMEAN(i)=MEANp-MEANn;
    DSD(i)=SDp-SDn;
    
    [pi,h] = ranksum(errorn,errorp); 
    if MEANp-MEANn>0; p(i)=pi; else p(i)=1+abs(pi);interfer(i)=1;end
end

[sortedp,indexp]=sort(p);
COSS=-log10(p);
COSS(COSS<0)=0;
toc;
%+++ output  %%%%%%%%%%%%%%%%
F.method=method;
F.time=toc/60;
F.N=N;
F.ratio=ratio;
F.Q=Q;
F.nLV=nLV;
F.NPE=error0;
F.PPE=error1;
F.DMEAN=DMEAN;
F.DSD=DSD;
F.interfer=interfer;
F.p=p;
F.COSS=COSS;
F.RankedVariable=indexp;
%+++ Save results





