function Black_White_Processing(Csv_folder, Black_Reference_file, Output_file)
% 功能：只有黑参考时的黑校正
% 处理公式：Post_Processing_data = Sample - Black

if exist(Output_file, 'file') ~= 0
    return;
end

save_dir = fileparts(Output_file);
if ~isempty(save_dir) && exist(save_dir, 'dir') == 0
    mkdir(save_dir);
end

Csv_files = dir(fullfile(Csv_folder, '*.csv'));
[~, idx] = sort({Csv_files.name});
Csv_files = Csv_files(idx);
if isempty(Csv_files)
    error('未找到待处理的光谱文件。');
end

Average_num = 1;
fin_data = [];
data = [];

for i = 1:length(Csv_files)
    filename = Csv_files(i).name;
    file_path = fullfile(Csv_folder, filename);
    new_data = local_read_spectrum_col(file_path);

    if isempty(data)
        data = zeros(length(new_data), 1);
    elseif length(new_data) ~= length(data)
        x_old = linspace(0, 1, length(new_data));
        x_new = linspace(0, 1, length(data));
        new_data = interp1(x_old, new_data, x_new, 'linear', 'extrap')';
    end

    data = data + new_data;
    if mod(i, Average_num) == 0
        data = data / Average_num;
        fin_data = cat(2, fin_data, data);
        data = zeros(size(data));
    end
end

Processing_data = fin_data';
Black_data = local_read_spectrum_col(Black_Reference_file);
if length(Black_data) ~= size(Processing_data, 2)
    x_old = linspace(0, 1, length(Black_data));
    x_new = linspace(0, 1, size(Processing_data, 2));
    Black_data = interp1(x_old, Black_data, x_new, 'linear', 'extrap')';
end

Post_Processing_data = Processing_data - Black_data';
writematrix(Post_Processing_data, Output_file);

fprintf('黑参考校正完成（只有黑参考，使用减黑公式）。\n');
fprintf('输出文件：%s\n', Output_file);
end

function spectrum_col = local_read_spectrum_col(file_path)
raw = readmatrix(file_path);
if isempty(raw)
    error('读取文件失败或无数值内容: %s', file_path);
end

if size(raw, 2) >= 2
    spectrum_col = raw(:, end);
else
    spectrum_col = raw(:);
end

spectrum_col = spectrum_col(~isnan(spectrum_col));
spectrum_col = spectrum_col(:);
end