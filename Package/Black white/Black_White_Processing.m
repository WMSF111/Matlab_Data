%% 黑白处理，数据合并函数
% 输入：Csv_folder：单个数据文件目录
% Black_White_folder：黑白光谱文件
% Black_White_Save_folder：黑白矫正处理后数据保存文件

function Black_White_Processing(Csv_folder, Black_White_folder, Black_White_Save_folder)
    % 判断是否存在该文件，如果不存在，则进行处理
    if exist(Black_White_Save_folder, 'file') == 0
        % 获取文件夹中所有CSV文件的名称
        Csv_files = dir(fullfile(Csv_folder, '*.csv'));

        % 对文件名称进行排序
        [~, idx] = sort({Csv_files.name});
        Csv_files = Csv_files(idx);

        % 循环读取每个CSV文件的内容
        fin_data = [];      % 存储数据
        Average_num = 3;    % 取平均个数
        data = [];

        for i = 1:length(Csv_files)
            filename = Csv_files(i).name;
            file_path = fullfile(Csv_folder, filename);

            % 兼容 1列/2列等格式：默认取最后一列作为光谱强度
            new_data = local_read_spectrum_col(file_path);

            if isempty(data)
                data = zeros(length(new_data), 1);
            elseif length(new_data) ~= length(data)
                % 长度不一致时做插值对齐
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

        % 转置，列表示光谱数值，行为样本
        Processing_data = fin_data';

        % 黑白矫正
        Black_White_data = local_read_spectrum_col(Black_White_folder);
        if length(Black_White_data) ~= size(Processing_data, 2)
            x_old = linspace(0, 1, length(Black_White_data));
            x_new = linspace(0, 1, size(Processing_data, 2));
            Black_White_data = interp1(x_old, Black_White_data, x_new, 'linear', 'extrap')';
        end

        Post_Processing_data = Processing_data ./ Black_White_data';
        xlswrite(Black_White_Save_folder, Post_Processing_data);
    end
end

function spectrum_col = local_read_spectrum_col(file_path)
    raw = xlsread(file_path);
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
