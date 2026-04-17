function [y_out, info] = normalize_plot_series(y, mode, ref_y)
% Normalize a series for plotting using a selectable method.

if nargin < 2 || isempty(mode)
    mode = 'minmax';
end
if nargin < 3 || isempty(ref_y)
    ref_y = y;
end

mode = lower(strtrim(string(mode)));
ref_y = ref_y(isfinite(ref_y));
if isempty(ref_y)
    ref_y = 0;
end

y_out = y;
info = struct();
info.mode = char(mode);
info.is_normalized = ~strcmp(info.mode, 'none');
info.y_label_suffix_zh = char([65288 26410 24402 19968 21270 65289]);
info.y_label_suffix_en = ' (Raw)';

switch info.mode
    case 'none'
        return;

    case 'minmax'
        lo = min(ref_y);
        hi = max(ref_y);
        if abs(hi - lo) < eps
            y_out = zeros(size(y));
        else
            y_out = (y - lo) ./ (hi - lo);
        end
        info.y_label_suffix_zh = char([65288 26368 22823 26368 23567 24402 19968 21270 65289]);
        info.y_label_suffix_en = ' (Min-Max)';

    case 'zscore'
        mu = mean(ref_y, 'omitnan');
        sigma = std(ref_y, 'omitnan');
        if ~isfinite(sigma) || sigma < eps
            y_out = zeros(size(y));
        else
            y_out = (y - mu) ./ sigma;
        end
        info.y_label_suffix_zh = char([65288 26631 20934 21270 65289]);
        info.y_label_suffix_en = ' (Z-Score)';

    case 'maxabs'
        scale = max(abs(ref_y));
        if ~isfinite(scale) || scale < eps
            y_out = zeros(size(y));
        else
            y_out = y ./ scale;
        end
        info.y_label_suffix_zh = char([65288 26368 22823 32477 23545 20540 24402 19968 21270 65289]);
        info.y_label_suffix_en = ' (MaxAbs)';

    case 'robust'
        medv = median(ref_y, 'omitnan');
        q1 = prctile(ref_y, 25);
        q3 = prctile(ref_y, 75);
        iqr_v = q3 - q1;
        if ~isfinite(iqr_v) || iqr_v < eps
            y_out = zeros(size(y));
        else
            y_out = (y - medv) ./ iqr_v;
        end
        info.y_label_suffix_zh = char([65288 31283 20581 26631 20934 21270 65289]);
        info.y_label_suffix_en = ' (Robust)';

    case 'l2'
        scale = norm(ref_y(:), 2);
        if ~isfinite(scale) || scale < eps
            y_out = zeros(size(y));
        else
            y_out = y ./ scale;
        end
        info.y_label_suffix_zh = char([65288 76 50 24402 19968 21270 65289]);
        info.y_label_suffix_en = ' (L2)';

    otherwise
        error('Unsupported normalize mode: %s', char(mode));
end
end
