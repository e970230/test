clc
clear
close all
tic
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

AA=[120 680 890 1100 1400 1500 2500];



rpm_680_size=size(rpm_680,1);
rpm_890_size=size(rpm_890,1);
rpm_1100_size=size(rpm_1100,1);
rpm_1400_size=size(rpm_1400,1);
rpm_1500_size=size(rpm_1500,1);
rpm_2500_size=size(rpm_2500,1);


%各轉速隨機數

rpm_120_rand=randperm(size(rpm_120,1));
rpm_680_rand=randperm(size(rpm_680,1));
rpm_890_rand=randperm(size(rpm_890,1));
rpm_1100_rand=randperm(size(rpm_1100,1));
rpm_1400_rand=randperm(size(rpm_1400,1));
rpm_1500_rand=randperm(size(rpm_1500,1));
rpm_2500_rand=randperm(size(rpm_2500,1));



randomly_selected_samples=100;




train_data=[rpm_120(rpm_120_rand(1:randomly_selected_samples),2:11);rpm_680(rpm_680_rand(1:randomly_selected_samples),2:11);rpm_890(rpm_890_rand(1:randomly_selected_samples),2:11);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),2:11);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),2:11);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),2:11);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),2:11)];
train_data_y=[rpm_120(rpm_120_rand(1:randomly_selected_samples),1);rpm_680(rpm_680_rand(1:randomly_selected_samples),1);rpm_890(rpm_890_rand(1:randomly_selected_samples),1);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),1);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),1);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),1);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),1)];
input_data=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),2:11);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),2:11);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),2:11);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),2:11);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),2:11);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),2:11);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),2:11)];
input_data_answer=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),1);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),1);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),1);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),1);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),1);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),1);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),1)];


% x=[1 1 1];

% 設定基因演算法的初始參數
PopulationSize=100;     %族群大小意旨染色體數目
FitnessLimit=1000;      %目標函數跳出門檻(此題為小於此值則演算法結束
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'PlotFcn',{'gaplotbestf','gaplotbestindiv','gaplotexpectation'},'FitnessLimit',FitnessLimit);
%'iter'顯示出每次跌代的詳細資訊
%'PlotFcn'畫圖指令
% gaplotbestf紀錄每次跌代中最佳答案的分數(fitness)
% gaplotbestindiv最佳解答(x)
% gaplotexpectation每次跌代的期望值


% 定義要優化的參數範圍
numVariables = 3; % 三個參數
lb = [1, 1, 1];  % 下限
ub = [40, 20, 20]; % 上限

% 定義訓練數據和驗證數據
trainData =train_data; % 訓練數據
trainLabels =train_data_y; % 訓練標籤
validData = input_data; % 驗證數據
validLabels = input_data_answer; % 驗證標籤


% fitness=RandomForestFitness(x, trainData, trainLabels, validData, validLabels);


%%

% 定義目標函數
fitnessFunction = @(x) RandomForestFitness(x, trainData, trainLabels, validData, validLabels);

% 使用基因演算法搜索最佳參數
[x,fval,exitflag,output,population,scores] = ga(fitnessFunction, numVariables, [], [], [], [], lb, ub, [], options);

%[x]解答
%[fval]分數
%[exitflag]結束運算的原因
%[output]輸出運算法的結構
%[population]最後一次跌代所有染色體的解答
%[scores]最後一次跌代所有染色體的分數


%%
% 最優參數
bestNumTrees = round(x(1));
bestMaxDepth = round(x(2));
bestMinLeafSize = round(x(3));

% 使用最優參數訓練最終的隨機森林模型
finalModel = TreeBagger(bestNumTrees, trainData, trainLabels, 'Method', 'regression', ...
    'MaxNumSplits', bestMaxDepth, 'MinLeafSize', bestMinLeafSize);
predictions = predict(finalModel, validData);

disp('隨機森林裡決策樹的數量')
disp(bestNumTrees);
disp('葉節點最小樣本數')
disp(bestMinLeafSize);
disp('每顆樹最大的分割次數')
disp(bestMaxDepth);



toc
%%

corrict_time=length(find(abs(predictions-validLabels)==0))

