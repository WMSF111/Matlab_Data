%% PCA降维函数
function [PCA_data] = PCA(data)
[~, score, latent, ~, ~, ~]=pca(data);
a=cumsum(latent)./sum(latent);   % 计算特征的累计贡献率
idx=find(a>0.99);     % 将特征的累计贡献率不小于0.9的维数作为PCA降维后特征的个数
k=idx(1);
PCA_data=score(:,1:k);   % 取转换后的矩阵score的前k列为PCA降维后特征
end
