clc
clear
close all


%% 初始設立條件
feature_dataset=load("feature_dataset_heavy.mat");
first_data=feature_dataset.feature_dataset;
% load("answer_for_ga.mat","answer_for_ga")
answer_for_ga=[114 32 5];


train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);

trainData =train_data_all; % 訓練數據
trainLabels =train_data_all_y; % 訓練標籤
validData = train_data_all; % 驗證數據
validLabels = train_data_all_y; % 驗證標籤

litera=size(answer_for_ga,1);

nov=[1 1 1 final_sample_input_answer];

% for tre=1:10
tic
ra_test=30;
for li=1:litera

    for i=1:ra_test
        input_numTrees = answer_for_ga(li,1);%森林中樹的數量
        input_MaxNumSplits=answer_for_ga(li,2);%每顆樹最大的分割次數
        input_MinLeafSize=answer_for_ga(li,3);%葉節點最小樣本數
        
        X=[input_numTrees,input_MaxNumSplits,input_MinLeafSize];
        fitness(li,i) = RandomForestFitness(X, trainData, trainLabels, validData, validLabels,nov);
        disp("第" + num2str(li) + "組" + "第" + num2str(i) + "次適性值")
        disp(fitness(li,i))
    end


end
%%
mean_fitness=mean(fitness,2);
var_fitness=var(fitness,0,2);
std_fitness=std(fitness,0,2);

disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林平均值" + num2str(mean_fitness))
disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林變異數" + num2str(var_fitness))
disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林標準差" + num2str(std_fitness))
toc
%%
unique(final_sample_input_answer)


disp('所選的特徵')
disp(final_sample_input_answer)

% end
% delete(gcp);
% true_answer=mse(input_data_answer)/size(input_data_answer,1);

% ture=mse_mean_answer;
% disp(ture)

