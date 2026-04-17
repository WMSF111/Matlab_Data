clc; clear; close all;
clear functions;
lang_mode = 'zh';
normalize_mode = 'minmax';


% result70 = predict_property_time_series(70, [], [], lang_mode, normalize_mode);
% result80 = predict_property_time_series(80, [], [], lang_mode, normalize_mode);
% result90 = predict_property_time_series(90, [], [], lang_mode, normalize_mode);
result_compare = compare_temperature_time_series([70 80 90], 0:20, [], lang_mode, normalize_mode);
