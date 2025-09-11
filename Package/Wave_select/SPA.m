function [SelectedW] = SPA(SpecCal,Winitial,totN)
% SpecCal 光谱矩阵（行为样品，列为波段）
% Winitial 起始波段
% totN 选择的波段总数
% SelectedW 最终选择的波段
[NoSp,Novab] = size(SpecCal);
Varibs = 1:Novab;
SelectedW = ones(1,totN);
Specj = SpecCal;  
Specn = SpecCal(:,Winitial);
SelectedW(1) = Winitial;
for n = 1:totN-1 %待确定变量数的循环
    litW =SelectedW(1:n);
    Jnotsel = setdiff(Varibs,litW);  %确定未映射变量
    APSpecj = zeros(1,length(Jnotsel));
    PSpecj = zeros(NoSp,Novab);
    stP = 1;
    for j = Jnotsel %未确定变量的循环
        PSpecj(:,j) = Specj(:,j) - (Specj(:,j)'*Specn)*Specn*(Specn'*Specn)^(-1);         
        APSpecj(stP) = norm(PSpecj(:,j));
        stP = stP+1;
    end
    SelectedW(n+1) = Jnotsel(APSpecj==max(APSpecj));
    Specn = SpecCal(:,SelectedW(n+1));
    Specj = PSpecj;
end
end