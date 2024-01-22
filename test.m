clc
clear
close all

% 创建包含字符串的cell数组
cellMatrix = {'0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2'};

% 使用str2double将cell数组中的字符串转换为数值
numericMatrix = str2double(cellMatrix);

% 显示结果
disp(numericMatrix);
