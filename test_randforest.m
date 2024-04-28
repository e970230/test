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

X=[153 34 5];

% fitness = RandomForestFitness(X, trainData, trainLabels, validData, validLabels);

for it=1:30

    fitness(it) = RandomForestFitnessBasic(X, trainData, trainLabels, validData, validLabels);
end

mean_fitness=mean(fitness,2);
var_fitness=var(fitness,0,2);
std_fitness=std(fitness,0,2);


disp('隨機森林裡決策樹的數量')
disp(X(1));
disp('每顆樹最大的分割次數')
disp(X(2)); 
disp('葉節點最小樣本數')
disp(X(3));
disp('歷代最佳分數')
disp(min(fitness));
disp("相同超參數重複建立" + num2str(it)+ "次隨機森林所得平均值" + num2str(mean_fitness))
disp("相同超參數重複建立" + num2str(it)+ "次隨機森林所得變異數" + num2str(var_fitness))
disp("相同超參數重複建立" + num2str(it)+ "次隨機森林所得標準差" + num2str(std_fitness))