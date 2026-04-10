% 功能：项目统一入口（推荐使用）
% 作用：
%   1) 自动加入项目路径
%   2) 支持按 all_csv_data.csv 的列头进行训练
%   3) 支持普通建模和“特征筛选+建模”两种模式
%
% 输入参数：
%   property_name : 目标属性名（如 'a*'、'Mrix'、'PH'）
%                   特殊值 'list' 用于查看可用列头
%   method_name   : 建模方法
%                   'spa' / 'cars' / 'pca' / 'rfe' / 'fs'
%                   其中 'fs' 表示走“特征筛选+建模”流程
%   preproc_mode  : 预处理模式
%                   'sg+msc'（默认）/ 'sg+snv' / 'sg+msc+snv' / 'sg' / 'none'
%   sg_order      : SG 多项式阶数（默认 3）
%   sg_window     : SG 窗口长度（默认 15）
%
% 完整使用示例：
%   % 0) 切到项目目录（建议）
%   cd('E:\study\algriothm\matlab\Data processing');
%   clear functions;
%
%   % 1) 查看 all_csv_data.csv 可用属性列
%   run_property_prediction('list');
%
%   % 2) 普通建模示例：预测 a*，方法 SPA，预处理 SG+MSC+SNV
%   run_property_prediction('a*','spa','sg+msc+snv',3,15);
%
%   % 3) 普通建模示例：预测 Mrix，方法 PCA，预处理 SG+MSC
%   run_property_prediction('Mrix','pca','sg+msc',3,15);
%
%   % 4) 特征筛选示例：预测 a*，启用特征筛选流程（corr_topk）
%   run_property_prediction('a*','fs','sg+msc+snv',3,15);
function run_property_prediction(property_name, method_name, preproc_mode, sg_order, sg_window)

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

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

if strcmpi(property_name, 'list')
    list_all_csv_headers();
    return;
end

if strcmpi(method_name, 'fs')
    process_by_all_csv_with_feature_selection(property_name, preproc_mode, sg_order, sg_window, 'corr_topk', 120);
else
    process_by_all_csv_header(property_name, method_name, preproc_mode, sg_order, sg_window);
end
end
