clc
clear
close all

tic
feature_dataset=load("feature_dataset_top30.mat");
first_data=feature_dataset.feature_top30.dataset;


%%

train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);

intcon=[1 2 3];
% 設定基因演算法的初始參數`
PopulationSize=30;     %族群大小意旨染色體數目
% FitnessLimit=5;      %目標函數跳出門檻(小於此值則演算法結束
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'Generations',1000,'CrossoverFraction', 0.5,'OutputFcn',@gaoutputfunction);
%'iter'顯示出每次疊代的詳細資訊
%'PlotFcn'畫圖指令
% gaplotbestf紀錄每次跌代中最佳答案的分數(fitness)
% gaplotbestindiv最佳解答(x)
% gaplotexpectation每次跌代的期望值
% Generations疊代次數
% OutputFun輸出資訊客製化
% CrossoverFraction更改交配率

% 定義要優化的參數範圍
numVariables = 3; % 三個參數(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數)
lb = [100, 5, 2];  % 下限
ub = [475, 100, 10]; % 上限



%%
%全部資料集和標籤
trainData =train_data_all; % 訓練數據
trainLabels =train_data_all_y; % 訓練標籤
validData = train_data_all; % 驗證數據
validLabels = train_data_all_y; % 驗證標籤


%%

% 定義目標函數
fitnessFunction = @(x) RandomForestFitness(x, trainData, trainLabels, validData, validLabels);

% 使用基因演算法搜索最佳參數
[x,fval,exitflag,output,population,scores] = ga(fitnessFunction, numVariables, [], [], [], [], lb, ub, [],intcon,options);

%[x]解答
%[fval]分數
%[exitflag]結束運算的原因
%[output]輸出運算法的結構
%[population]最後一次跌代所有染色體的解答
%[scores]最後一次跌代所有染色體的分數
%Func-count調動目標函數的次數
%stall generations疊代沒有優化的次數

% delete(gcp);

%%
answer=[];
final_answer=[];
for pl=1:size(Population_answer,3)
    [answer(pl),answer_index(pl)]=min(Population_answer(:,5,pl));
    final_answer(pl,:)=Population_answer(answer_index(pl),1:4,pl);
end

[final_answer_ture,final_answer_index]=min(final_answer(:,4));
final_answer_input=final_answer(final_answer_index,:);

%%
% 最優參數
bestNumTrees = round(final_answer_input(1));
bestMaxDepth = round(final_answer_input(2));
bestMinLeafSize = round(final_answer_input(3));

% 使用最優參數訓練最終的隨機森林模型
finalModel = TreeBagger(bestNumTrees, trainData, trainLabels, 'Method', 'regression', ...
    'MaxNumSplits', bestMaxDepth, 'MinLeafSize', bestMinLeafSize);
predictions = predict(finalModel, validData);

disp('隨機森林裡決策樹的數量')
disp(bestNumTrees);
disp('每顆樹最大的分割次數')
disp(bestMaxDepth); 
disp('葉節點最小樣本數')
disp(bestMinLeafSize);
disp('歷代最佳分數')
disp(final_answer_ture);




toc
%%
figure(2)
plot(1:size(final_answer,1),final_answer(:,4));
xlabel('疊代次數')
ylabel('每次疊代最佳MSE')

%%
corrict_time=length(find(abs(predictions-validLabels)==0))
round_predictions=round(predictions);
