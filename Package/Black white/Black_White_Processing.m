%% 黑白处理，数据合并函数
% 输入：Csv_folder： 单个数据文件目录
% Black_White_folder：黑白光谱文件
% Black_White_Save_folder：黑白矫正处理后数据保存文件

function  Black_White_Processing(Csv_folder,Black_White_folder,Black_White_Save_folder)
    %判断是否存在该文件，如果不存在，则进行处理
    if exist(Black_White_Save_folder, 'file') == 0
%     if exist(Save_folder, 'file') == 0
        % 获取文件夹中所有CSV文件的名称
        Csv_files = dir(fullfile(Csv_folder, '*.csv'));

        % 对文件名称进行排序，按名称从大到小的顺序排列，仅限于文件命名是数字的情况
        [~, idx] = sort({Csv_files.name});
        Csv_files = Csv_files(idx);

        % 循环读取每个CSV文件的内容
        fin_data = [];%存储数据
        data = zeros(256,1);
        Average_num = 3;%取平均个数

        for i = 1:length(Csv_files)
            % 获取当前CSV文件的名称
            filename = Csv_files(i).name;
            % 使用xlsread函数读取CSV文件内容，存储在一个矩阵中
            new_data = xlsread(fullfile(Csv_folder, filename));%只读取数据
            data = data + new_data;
            if( mod(i,3) == 0)
                data = data/Average_num;%数据求平均
                fin_data = cat(2,fin_data,data);%将每次读取的数据拼接到一起，第一维度
                data = zeros(256,1);
            end
        end
        %转置，列表示光谱数值，行为样本
        fin_data = fin_data';
        %黑白矫正
        Processing_data = fin_data;
        Black_White_data = xlsread(Black_White_folder);
        Post_Processing_data = Processing_data./Black_White_data';
        xlswrite(Black_White_Save_folder,Post_Processing_data); %保存数据
    end
end
