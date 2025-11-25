function [ypred,RMSEP]=plsval(plsmodel,Xtest,ytest,nLV)
%+++ 计算测试集上的预测误差
%+++ plsmodel：从本目录中的 pls.m 函数获得的结构数据；
%+++ Xtest：测试样本
%+++ ytest：测试样本的 y 值（对于“真正”新的样本，没有 y 值可用）
%+++ nLV：校准模型的潜变量数量。

if nargin<4;nLV=size(plsmodel.X_scores,2);end;

Xtest=[Xtest ones(size(Xtest,1),1)]; 
ypred=Xtest*plsmodel.regcoef_original_all(:,nLV);
RMSEP=sqrt(sumsqr(ypred-ytest)/length(ytest));






