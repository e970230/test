clc
clear
close all


%% 初始設立條件
feature_dataset=load("feature_dataset_heavy.mat");
first_data=feature_dataset.feature_dataset;
bit2int = @(bits) sum(bits .* 2.^(length(bits)-1:-1:0));
rpm_120=first_data(find(first_data==120),:);
rpm_680=first_data(find(first_data==680),:);
rpm_890=first_data(find(first_data==890),:);
rpm_1100=first_data(find(first_data==1100),:);
rpm_1400=first_data(find(first_data==1400),:);
rpm_1500=first_data(find(first_data==1500),:);
rpm_2500=first_data(find(first_data==2500),:);

sampled_data=100;

train_data=[rpm_120(1:sampled_data,2:end);rpm_680(1:sampled_data,2:end);rpm_890(1:sampled_data,2:end);rpm_1100(1:sampled_data,2:end);rpm_1400(1:sampled_data,2:end);rpm_1500(1:sampled_data,2:end);rpm_2500(1:sampled_data,2:end)];
train_data_y=[rpm_120(1:sampled_data,1);rpm_680(1:sampled_data,1);rpm_890(1:sampled_data,1);rpm_1100(1:sampled_data,1);rpm_1400(1:sampled_data,1);rpm_1500(1:sampled_data,1);rpm_2500(1:sampled_data,1)];
input_data=[rpm_120(sampled_data+1:end,2:end);rpm_680(sampled_data+1:end,2:end);rpm_890(sampled_data+1:end,2:end);rpm_1100(sampled_data+1:end,2:end);rpm_1400(sampled_data+1:end,2:end);rpm_1500(sampled_data+1:end,2:end);rpm_2500(sampled_data+1:end,2:end)];
input_data_answer=[rpm_120(sampled_data+1:end,1);rpm_680(sampled_data+1:end,1);rpm_890(sampled_data+1:end,1);rpm_1100(sampled_data+1:end,1);rpm_1400(sampled_data+1:end,1);rpm_1500(sampled_data+1:end,1);rpm_2500(sampled_data+1:end,1)];


input_numTrees = 4;%森林中樹的數量
input_MinLeafSize=4;%葉節點最小樣本數
input_MaxNumSplits=4;%每顆樹最大的分割次數
insurance_value=0.00001; %以防再答案完全吻合時分母為0的保險

treeBaggerModel = TreeBagger(input_numTrees, train_data, train_data_y,'Method','classification', ...
        'MinLeafSize',input_MinLeafSize,'MaxNumSplits',input_MaxNumSplits);
predictions = predict(treeBaggerModel, input_data);
answer_prediceions=str2double(predictions);%由於輸出的答案不是數值是字串，所以將字串轉數值

score_temporary_storage=((abs(answer_prediceions-input_data_answer)));
error_time=(length(find(score_temporary_storage)));
score=1/(error_time+insurance_value);