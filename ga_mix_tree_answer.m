clc
clear
close all

%% 網格搜索法

feature_dataset=load("feature_dataset_top30.mat");
first_data=feature_dataset.feature_top30.dataset;

train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);

trainData =train_data_all; % 訓練數據
trainLabels =train_data_all_y; % 訓練標籤
validData = train_data_all; % 驗證數據
validLabels = train_data_all_y; % 驗證標籤

x1_range=[10 200];
x2_range=[5 50];
x3_range=[5 10];
%%
tic
for i=x1_range(1):x1_range(2)
    for y=x2_range(1):x2_range(2)
        for z=x3_range(1):x3_range(2)
            x=[i y z];
            fitness(i-x1_range(1)+1,y-x2_range(1)+1,z-x3_range(1)+1) =RandomForestFitnessBasic(x, trainData, trainLabels, validData, validLabels);
            disp(['隨機森林裡決策樹的數量 = ', num2str(i), ', 每顆樹最大的分割次數 = ', num2str(y), ', 葉節點最小樣本數 ', num2str(z)]);
            disp(['適性值分數 = ' , num2str(fitness(i-x1_range(1)+1,y-x2_range(1)+1,z-x3_range(1)+1))])
        end
    end
end

 toc      
%%

[fitness_x2range_answer,fitness_answer_x2range_index]=min(fitness(:,:,1),[],2);
[a,a_index]=min(fitness_x2range_answer);
answer=fitness(a_index,fitness_answer_x2range_index(a_index),1)
disp(" 隨機森林裡決策樹的數量 "+num2str(a_index+x1_range(1))+" 每顆樹最大的分割次數 "+num2str(fitness_answer_x2range_index(a_index)+x2_range(1))+" 葉節點最小樣本數 "+num2str(x3_range(1)))