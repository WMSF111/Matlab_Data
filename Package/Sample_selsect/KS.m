%% Kennard-Stone algorithm函数
function [selectedsamplenumbers, remainingsamplenumbers] = KS(X,k)
% 使用Kennard-Stone algorithm选择样本
% --- 输入 ---
% X : 输入数据
% k : 选择样本个数
% --- 输出 ---
% selectedsamplenumbers : 选择的样本（训练数据）
% remainingsamplenumbers : 没有选择的样本 (测试数据)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

selectedsamplenumbers = zeros( 1, k);
remainingsamplenumbers = 1 : size( X, 1);
originalX = X;

[~, selectedsamplenumbers(1) ] = max(pdist2(X, mean(X)));
X( selectedsamplenumbers(1), :) = [];
remainingsamplenumbers(selectedsamplenumbers(1)) = [];

for iteration = 1 : k-1
    if iteration == 1
        [~, selectedsamplenumber] = max( pdist2( originalX(selectedsamplenumbers(1:iteration),:), X) );
    else
        [~, selectedsamplenumber] = max( min( pdist2( originalX(selectedsamplenumbers(1:iteration),:), X) ) );
    end
	selectedsamplenumbers(iteration+1) = remainingsamplenumbers(selectedsamplenumber);
    X( selectedsamplenumber, :) = [];
    remainingsamplenumbers(selectedsamplenumber) = [];
end
end



