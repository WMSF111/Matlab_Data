function [feat_row, feat_labels, feature_modes] = extract_e_nose_features(data_matrix, sensor_names, feature_modes)
% Extract selectable E-nose features for each sensor column.

if nargin < 3 || isempty(feature_modes)
    feature_modes = {'mean'};
end

feature_modes = local_normalize_feature_modes(feature_modes);
sensor_names = cellstr(string(sensor_names(:)'));
data_matrix = double(data_matrix);

feat_row = [];
feat_labels = {};

for i = 1:numel(feature_modes)
    mode_name = feature_modes{i};
    feat_vals = local_compute_feature(data_matrix, mode_name);
    feat_row = [feat_row, feat_vals]; %#ok<AGROW>
    feat_labels = [feat_labels, strcat(sensor_names, "_", mode_name)]; %#ok<AGROW>
end
end

function feature_modes = local_normalize_feature_modes(feature_modes)
if ischar(feature_modes) || (isstring(feature_modes) && isscalar(feature_modes))
    tokens = regexp(char(feature_modes), '[,+/|;\s]+', 'split');
    feature_modes = tokens(~cellfun(@isempty, tokens));
elseif isstring(feature_modes)
    feature_modes = cellstr(feature_modes(:)');
elseif ~iscell(feature_modes)
    error('feature_modes must be char, string, or cell array.');
end

feature_modes = cellfun(@(x) lower(strtrim(char(x))), feature_modes, 'UniformOutput', false);
feature_modes = feature_modes(~cellfun(@isempty, feature_modes));

if isempty(feature_modes)
    feature_modes = {'mean'};
end

if numel(feature_modes) == 1 && strcmp(feature_modes{1}, 'all')
    feature_modes = {'mean','max','min','auc','sum','med','mode','std','slope','cv'};
end

valid_modes = {'mean','max','min','auc','sum','med','mode','std','slope','cv'};
bad_modes = setdiff(unique(feature_modes), valid_modes);
if ~isempty(bad_modes)
    error('Unsupported E-nose feature mode: %s', strjoin(bad_modes, ', '));
end

[~, ia] = unique(feature_modes, 'stable');
feature_modes = feature_modes(sort(ia));
end

function feat_vals = local_compute_feature(data_matrix, mode_name)
switch mode_name
    case 'mean'
        feat_vals = mean(data_matrix, 1, 'omitnan');
    case 'max'
        feat_vals = max(data_matrix, [], 1);
    case 'min'
        feat_vals = min(data_matrix, [], 1);
    case {'auc', 'sum'}
        feat_vals = sum(data_matrix, 1, 'omitnan');
    case 'med'
        feat_vals = median(data_matrix, 1, 'omitnan');
    case 'mode'
        feat_vals = mode(data_matrix, 1);
    case 'std'
        feat_vals = std(data_matrix, 0, 1, 'omitnan');
    case 'slope'
        if size(data_matrix, 1) < 2
            feat_vals = zeros(1, size(data_matrix, 2));
        else
            feat_vals = mean(diff(data_matrix, 1, 1), 1, 'omitnan');
        end
    case 'cv'
        mu = mean(data_matrix, 1, 'omitnan');
        sigma = std(data_matrix, 0, 1, 'omitnan');
        feat_vals = sigma ./ (mu + eps);
    otherwise
        error('Unsupported feature mode: %s', mode_name);
end
end
