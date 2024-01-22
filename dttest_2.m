clc
clear
close all
frist_data=load("PCA_test_data.txt","-ascii");

input_data=[frist_data(46:50,1:4);frist_data(96:100,1:4);frist_data(146:150,1:4)];

%樹的數目
numTrees = 20;

%先建立一個與樹的數目相同的cell矩陣來存放等等tree的模型資料
tree = cell(1, numTrees);

%先隨機從中選出隨機50個樣本進行訓練
%生成一個隨機產生的配對編號
startValue = 1;
endValue = size(frist_data,1);
train_max_Value=50;

% 創建回歸模型
for i = 1:numTrees
    random_Value = randperm(endValue - startValue + 1) + startValue - 1; % 生成不重复的隨機配對編號
    train_data=[frist_data(random_Value(1:train_max_Value),1:4)];
    train_data_y=[frist_data(random_Value(1:train_max_Value),5)];
    tree{i} = fitctree(train_data, train_data_y,"PredictorNames",{'萼片_長' '萼片_寬' '花瓣_長' '花瓣_寬'}, ...
    'MinParentSize',2,'MaxNumSplits',5,'MinLeafSize',1);
    X_new = input_data;
    y_pred(i,:) = predict(tree{i}, X_new);%將輸入資料帶進當下創建的迴歸樹裡
end
%找到所有樹裡出現最多次的答案
for t=1:size(y_pred,2)
    y_pred_ture(t) = mode(y_pred(:,t));
end

%顯示每顆迴歸樹的模型
% for e=1:numTrees
%     view(tree{e}, 'Mode', 'graph')
% end