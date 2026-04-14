% ==========================================================
% [HOT] 自动找 "进入平稳" 的转折点（可调节误差）
% x:时间 y:拟合曲线数据 tol:误差（自己调，例：0.01, 0.005）
% ==========================================================
function [x_stable, idx_stable] = findStablePoint(x, y, tol)
    n = length(y);
    % 从最后往前找
    for idx = n-10 : -1 : 1  % 最后10点防止噪声
        % 计算当前点到终点的波动
        range = max(y(idx:end)) - min(y(idx:end));
        if range < tol
            x_stable = x(idx);
            idx_stable = idx;
            return;
        end
    end
    x_stable = x(end);
    idx_stable = n;
end