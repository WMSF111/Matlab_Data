clc;
clear;
close;

% 黑白矫正
%% 保存文件夹
folder = 'D:\\Desktop';   %本文件夹存储位置
black_and_white_file_name = '1-0.csv'; %黑白矫正数据文件名

%csv文件未处理数据所在目录
Csv_folder =sprintf('%s\\Data processing\\data\\NIR',folder);
%黑白光谱文件
Black_White_folder =sprintf('%s\\Data processing\\data\\Black white\\%s',folder,black_and_white_file_name);
%黑白矫正处理后数据保存文件
Black_White_Save_folder = sprintf('%s\\Data processing\\Result\\Black_White\\post_processing_data.xlsx',folder);

%% 黑白矫正
Black_White_Processing(Csv_folder,Black_White_folder,Black_White_Save_folder);



 





