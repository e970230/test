clc
clear
close all

tic
data=load("feature_dataset_heavy.mat");
first_data=data.feature_dataset;
train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);

%全部資料集和標籤
trainData =train_data_all; % 訓練數據
trainLabels =train_data_all_y; % 訓練標籤
validData = train_data_all; % 驗證數據
validLabels = train_data_all_y; % 驗證標籤

% feature_dataset=load("feature_dataset_top30.mat");
% first_data=feature_dataset.feature_top30.dataset;

%%

% 設定基因演算法的初始參數
PopulationSize=10;     %族群大小意旨染色體數目
% FitnessLimit=5;      %目標函數跳出門檻(小於此值則演算法結束
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'Generations',10,'CrossoverFraction', 0.7,'OutputFcn',@gaoutputfunction);
%'Display','iter'顯示出每次疊代的詳細資訊
%'PlotFcn'畫圖指令
% gaplotbestf紀錄每次跌代中最佳答案的分數(fitness)
% gaplotbestindiv最佳解答(x)
% gaplotexpectation每次跌代的期望值
% Generations疊代次數
% OutputFun輸出資訊客製化
% CrossoverFraction更改交配率

Y_sample=30;    %選取要得樣本數量
% 定義要優化的參數範圍
numVariables = 3+Y_sample; % 三個參數(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數)加上選取的樣本數量
intcon=1:numVariables;  %整數變數的數量


lb_sample=ones(1,Y_sample); %每個選取樣本的下限
ub_sample=size(train_data_all,2)*ones(1,Y_sample); %每個選取樣本的上限
lb_input=[100 5 2];         %超參數猜值下限
ub_input=[1000 100 10];     %超參數猜值上限
lb = [lb_input lb_sample];  % 下限
ub = [ub_input ub_sample]; % 上限




%%

% 定義目標函數，x裡包含所有的猜值(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數、選取的樣本
fitnessFunction = @(x) RandomForestFitness(x, trainData, trainLabels, validData, validLabels,x);

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


%% 撈出最佳解的資訊
answer=[];
final_answer=[];
for pl=1:size(Population_answer,3)  %將最佳解(適性值、最佳的超參數配合)從相對應的疊代裡撈出來
    [answer(pl),answer_index(pl)]=min(Population_answer(:,end,pl));
    final_answer(pl,:)=Population_answer(answer_index(pl),[1:3 end],pl);
end

[final_answer_ture,final_answer_index]=min(final_answer(:,4));  %根據適性值把最好的超參數設定連同適性值分數找出來
final_answer_input=final_answer(final_answer_index,:);          %存成另一個矩陣
sample=Population_answer(:,:,final_answer_index);               %找出(選取特徵)出現過最佳解的疊代資訊
[final_sample,final_sample_index]=min(sample(:,end));           %在此次疊代理將最佳選取特徵的資訊找出來
final_sample_input_answer=sample(final_sample_index,4:end-2);   %存成另一個答案矩陣
%% 將得到最優解的超參數進行驗證

ra_test=30; %重複驗證次數
litera=size(final_answer_input,1);  %需驗證的答案組數
nov=[1 1 1 final_sample_input_answer];  %前面多了無關緊要的3個數字，只是因為nov在讀取時是從第四個開始
for li=1:litera

    for i=1:ra_test
        input_numTrees = final_answer_input(1);%森林中樹的數量
        input_MaxNumSplits= final_answer_input(2);%每顆樹最大的分割次數
        input_MinLeafSize= final_answer_input(3);%葉節點最小樣本數
        X=[input_numTrees,input_MaxNumSplits,input_MinLeafSize];
        %將完全一樣的條件建立多次的森林得到浮動的適性值
        fitness(li,i) = RandomForestFitness(X, trainData, trainLabels, validData, validLabels,nov);
        disp("第" + num2str(li) + "組" + "第" + num2str(i) + "次適性值")
        disp(fitness(li,i))
    end


end
%在利用平均數、變異數、標準差等方法來評判此種超參數設定答案是否會浮動過大
%進而避免導致出現人品爆棚的答案，誤以為是最佳解
mean_fitness=mean(fitness,2);
var_fitness=var(fitness,0,2);
std_fitness=std(fitness,0,2);


disp('隨機森林裡決策樹的數量')
disp(input_numTrees);
disp('每顆樹最大的分割次數')
disp(input_MaxNumSplits); 
disp('葉節點最小樣本數')
disp(input_MinLeafSize);
disp('歷代最佳分數')
disp(final_answer_ture);
disp('選擇的特徵')
disp(sort(final_sample_input_answer));
disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得平均值" + num2str(mean_fitness))
disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得變異數" + num2str(var_fitness))
disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得標準差" + num2str(std_fitness))



toc
%%
figure(2)
plot(1:size(final_answer,1),final_answer(:,end));
title('適性值疊代過程')
xlabel('疊代次數')
ylabel('每次疊代最佳MSE')



