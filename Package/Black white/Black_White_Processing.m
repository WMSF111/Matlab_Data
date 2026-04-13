function Black_White_Processing(Csv_folder, Black_White_folder, Black_White_Save_folder)
    if exist(Black_White_Save_folder, 'file') == 0
        save_dir = fileparts(Black_White_Save_folder);
        if ~isempty(save_dir) && exist(save_dir, 'dir') == 0
            mkdir(save_dir);
        end

        Csv_files = dir(fullfile(Csv_folder, '*.csv'));
        [~, idx] = sort({Csv_files.name});
        Csv_files = Csv_files(idx);

        fin_data = [];
        Average_num = 1;
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

        Black_White_data = local_read_spectrum_col(Black_White_folder);
        if length(Black_White_data) ~= size(Processing_data, 2)
            x_old = linspace(0, 1, length(Black_White_data));
            x_new = linspace(0, 1, size(Processing_data, 2));
            Black_White_data = interp1(x_old, Black_White_data, x_new, 'linear', 'extrap')';
        end

        Post_Processing_data = Processing_data ./ Black_White_data';
        writematrix(Post_Processing_data, Black_White_Save_folder);
    end
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
