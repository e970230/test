clc
clear
clear Population_answer
close all
tic
feature_dataset=load("feature_dataset_top30.mat");
first_data=feature_dataset.feature_top30.dataset;
bit2int = @(bits) sum(bits .* 2.^(length(bits)-1:-1:0));

% 創建一個並行計算池，指定使用本地處理器的多個核心
% parpool('local', 5); % 這裡4代表使用4個核心，可以根據需要調整
% rpm_120=first_data(find(first_data==120),:);
% rpm_680=first_data(find(first_data==680),:);
% rpm_890=first_data(find(first_data==890),:);
% rpm_1100=first_data(find(first_data==1100),:);
% rpm_1400=first_data(find(first_data==1400),:);
% rpm_1500=first_data(find(first_data==1500),:);
% rpm_2500=first_data(find(first_data==2500),:);
% 
% 
% 
% 
% 
% rpm_680_size=size(rpm_680,1);
% rpm_890_size=size(rpm_890,1);
% rpm_1100_size=size(rpm_1100,1);
% rpm_1400_size=size(rpm_1400,1);
% rpm_1500_size=size(rpm_1500,1);
% rpm_2500_size=size(rpm_2500,1);
% 
% 
% %各轉速隨機數
% 
% rpm_120_rand=randperm(size(rpm_120,1));
% rpm_680_rand=randperm(size(rpm_680,1));
% rpm_890_rand=randperm(size(rpm_890,1));
% rpm_1100_rand=randperm(size(rpm_1100,1));
% rpm_1400_rand=randperm(size(rpm_1400,1));
% rpm_1500_rand=randperm(size(rpm_1500,1));
% rpm_2500_rand=randperm(size(rpm_2500,1));
% 
% 
% 
% randomly_selected_samples=100;
% 
% 
% 
% 
% train_data=[rpm_120(rpm_120_rand(1:randomly_selected_samples),2:11);rpm_680(rpm_680_rand(1:randomly_selected_samples),2:11);rpm_890(rpm_890_rand(1:randomly_selected_samples),2:11);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),2:11);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),2:11);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),2:11);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),2:11)];
% train_data_y=[rpm_120(rpm_120_rand(1:randomly_selected_samples),1);rpm_680(rpm_680_rand(1:randomly_selected_samples),1);rpm_890(rpm_890_rand(1:randomly_selected_samples),1);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),1);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),1);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),1);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),1)];
% input_data=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),2:11);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),2:11);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),2:11);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),2:11);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),2:11);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),2:11);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),2:11)];
% input_data_answer=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),1);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),1);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),1);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),1);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),1);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),1);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),1)];


%%

train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);


% 設定基因演算法的初始參數`
PopulationSize=20;     %族群大小意旨染色體數目
% FitnessLimit=5;      %目標函數跳出門檻(小於此值則演算法結束
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'Generations',1000,'OutputFcn',@gaoutputfunction);
%'iter'顯示出每次跌代的詳細資訊
%'PlotFcn'畫圖指令
% gaplotbestf紀錄每次跌代中最佳答案的分數(fitness)
% gaplotbestindiv最佳解答(x)
% gaplotexpectation每次跌代的期望值
% Generations跌代次數
% OutputFun輸出資訊客製化


% 定義要優化的參數範圍
numVariables = 3; % 三個參數(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數)
lb = [100, 5, 2];  % 下限
ub = [1000, 100, 10]; % 上限

% 定義訓練數據和驗證數據
% trainData =train_data; % 訓練數據
% trainLabels =train_data_y; % 訓練標籤
% validData = input_data; % 驗證數據
% validLabels = input_data_answer; % 驗證標籤


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
[x,fval,exitflag,output,population,scores] = ga(fitnessFunction, numVariables, [], [], [], [], lb, ub, [], options);

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
