function [SelectedW] = SPA(SpecCal,Winitial,totN)
% SpecCal 光谱矩阵（行=样品，列=波段）
% Winitial 起始波段
% totN 选择的波段总数
% SelectedW 最终选择的波段

[NoSp, Novab] = size(SpecCal);
Varibs = 1:Novab;
totN = min(totN, Novab);
SelectedW = ones(1, totN);
Specj = SpecCal;
Specn = SpecCal(:, Winitial);
SelectedW(1) = Winitial;

for n = 1:totN-1
    litW = SelectedW(1:n);
    Jnotsel = setdiff(Varibs, litW);
    APSpecj = zeros(1, length(Jnotsel));
    PSpecj = zeros(NoSp, Novab);
    stP = 1;

    denom = (Specn' * Specn);
    if abs(denom) < 1e-12
        denom = 1;
    end

    for j = Jnotsel
        PSpecj(:, j) = Specj(:, j) - ((Specj(:, j)' * Specn) / denom) * Specn;
        APSpecj(stP) = norm(PSpecj(:, j));
        stP = stP + 1;
    end

    [~, idx_max] = max(APSpecj);
    SelectedW(n + 1) = Jnotsel(idx_max(1));
    Specn = SpecCal(:, SelectedW(n + 1));
    Specj = PSpecj;
end
end
