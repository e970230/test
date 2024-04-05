clc
clear
close all

feature_dataset=load("feature_dataset_top30.mat");
first_data=feature_dataset.feature_top30.dataset;

train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);

trainData =train_data_all; % 訓練數據
trainLabels =train_data_all_y; % 訓練標籤
validData = train_data_all; % 驗證數據
validLabels = train_data_all_y; % 驗證標籤

X=[100 10 2];

% fitness = RandomForestFitness(X, trainData, trainLabels, validData, validLabels);

fitness = RandomForestFitness(X, trainData(:,2), trainLabels(:,1), validData(:,1), validLabels(:,1));