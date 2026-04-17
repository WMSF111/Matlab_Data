% 功能：项目统一入口（推荐使用）
% 作用：
%   1) 自动加入项目路径
%   2) 支持按 all_csv_data.csv 的列头进行训练
%   3) 支持普通建模、仅特征筛选、特征筛选+建模三种模式
%
% 输入参数：
%   property_name : 目标属性名（如 'a*'、'Mrix'、'PH'）
%                   特殊值 'list' 用于查看可用列头
%   method_name   : 运行模式
%                   'spa' / 'cars' / 'pca' / 'rfe'
%                   'select' 表示仅做特征筛选
%                   'fs' 表示“特征筛选+建模”
%   preproc_mode  : 预处理模式
%                   'sg+msc'（默认）/ 'sg+snv' / 'sg+msc+snv' / 'sg' / 'none'
%   sg_order      : SG 多项式阶数（默认 3）
%   sg_window     : SG 窗口长度（默认 15）
%   varargin{1}   : 特征筛选算法（仅当 method_name='select' 或 'fs' 时有效）
%                   'corr_topk' / 'pca' / 'cars' / 'spa'
%   varargin{2}   : 特征筛选参数
%                   corr_topk/pca/spa 时表示特征数
%                   cars 时表示 CARS 采样次数
%
% 特征筛选算法说明：
%   pca
%     - 无监督训练前降维/筛选，和训练耦合最弱
%   corr_topk
%     - 监督式训练前筛选，利用特征与 y 的相关性排序
%   spa
%     - 偏训练前筛选，主要减少特征冗余和共线性
%   cars
%     - 建模驱动筛选，和训练最密切相关
%
% 完整使用示例：
%   % 1) 查看 all_csv_data.csv 可用属性列
%   run_property_prediction('list');
%
%   % 2) 普通建模示例：预测 a*，方法 SPA，预处理 SG+MSC+SNV
%   run_property_prediction('a*','spa','sg+msc+snv',3,15);
%
%   % 3) 仅特征筛选示例：预测 a*，使用 PCA 先筛 40 个特征
%   run_property_prediction('a*','select','sg+msc',3,15,'pca',40);
%
%   % 4) 特征筛选+建模示例：预测 a*，使用 CARS 做筛选
%   run_property_prediction('a*','fs','sg+msc+snv',3,15,'cars',30);
function run_property_prediction(property_name, method_name, preproc_mode, sg_order, sg_window, varargin)

if nargin < 1 || isempty(property_name)
    error('请提供属性名，例如：run_property_prediction(''a*'')');
end
if nargin < 2 || isempty(method_name)
    method_name = 'spa';
end
if nargin < 3 || isempty(preproc_mode)
    preproc_mode = 'sg+msc';
end
if nargin < 4 || isempty(sg_order)
    sg_order = 3;
end
if nargin < 5 || isempty(sg_window)
    sg_window = 15;
end

fs_method = 'corr_topk';
fs_param = [];
if numel(varargin) >= 1 && ~isempty(varargin{1})
    fs_method = varargin{1};
end
if numel(varargin) >= 2
    fs_param = varargin{2};
end

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(project_root));

if strcmpi(property_name, 'list')
    list_all_csv_headers();
    return;
end

switch lower(method_name)
    case 'select'
        process_by_all_csv_feature_selection_only(property_name, preproc_mode, sg_order, sg_window, fs_method, fs_param);
    case 'fs'
        process_by_all_csv_with_feature_selection(property_name, preproc_mode, sg_order, sg_window, fs_method, fs_param);
    otherwise
        process_by_all_csv_header(property_name, method_name, preproc_mode, sg_order, sg_window);
end
end
