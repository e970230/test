function[] = ga_mix_tree_Fnc(ga_input,numFeats,lb_input,ub_input ,trainData, trainLabels, validData, validLabels)

% 透過基因演算法(GA)對迴歸型(regression)隨機森林(RF)的超參數(樹數目, 每棵樹最大的分枝次數, 
% 葉節點最小樣本數)與特徵選擇進行優化; 並根據所給定的訓練數據集(trainData)及對應的標籤(trainLabels),
% 搭配此三項最優超參數建立RF預測模型, 再透過驗證數據集(validData)及其對應的標籤(validLabels)
% 計算該RF模型的預測準確度(MSE), 並以MSE作為GA疊代過程中的目標函數


% input
%----------------------------------------------------
% ga_input: GA參數設定; 維度=1*3, 依序為 族群大小, 疊代次數上限, 交配率
% numFeats: 欲由GA挑選的特徵數量 
%           (numFeats=0: 不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數)
% lb_input: 欲優化的超參數的搜索範圍的下限; 維度=1*欲優化的超參數數目
% ub_input: 欲優化的超參數的搜索範圍的上限; 維度=1*欲優化的超參數數目
% trainData: RF的訓練數據集; 維度=sample數*特徵數
% trainLabels: trainData對應的標籤(Label); 維度=sample數*1
% validData:  RF的驗證數據集; 維度=sample數*特徵數
% validLabels: validData對應的標籤(Label); 維度=sample數*1


% output
%----------------------------------------------------



% 使用子程式: RandomForestFitnessBasic.m, RandomForestFitness.m,
%            gaoutputfunction.m


%last modification: 2024/05/07



% 設定基因演算法(GA)的參數
PopulationSize=ga_input(1);     %族群大小 (染色體數目)
Generations=ga_input(2);        %疊代次數上限
CrossoverFraction=ga_input(3);  %交配率


options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'Generations',Generations,'CrossoverFraction', CrossoverFraction,'OutputFcn',@gaoutputfunction);
% ga的'OutputFcn'設定了自訂的gaoutputfunction.m, ga將於每次疊代時呼叫並執行此function



% 定義要優化的參數範圍
numParas=3;    %欲以GA優化的超參數數目 (因為是限定用於RF, 故超參數為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數等三種
numVariables = numParas+numFeats; % 要讓GA進行搜索的項目總數(GA的設計變數的數量) = 欲優化的3個超參數加上欲由GA挑選的特徵數目
intcon=1:numVariables;  %整數變數的數量


lb_sample=ones(1,numFeats); %欲由GA挑選的特徵的編號的最低數值
ub_sample=size(trainData,2)*ones(1,numFeats); %欲由GA挑選的特徵的編號的最高數值
lb = [lb_input lb_sample];  %每個設計變數的範圍(下限)
ub = [ub_input ub_sample];  %每個設計變數的範圍(上限)



if numFeats==0  %不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數

    % x為ga.m所找到的解 (包含:樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)
    % 根據x所提供的RF超參數及訓練集(trainData)與對應的標籤(trainLabels), 
    % 建立迴歸RF模型, 再透過驗證集(validData)及其對應的標籤(validLabels)計算
    % 該RF模型的預測值的MSE,並作為ga的目標函數(fitnessFunction)
    fitnessFunction = @(x) RandomForestFitnessBasic(x, trainData, trainLabels, validData, validLabels);
    disp("演算模式: 採計所有特徵")

else
    
    % x為ga.m所找到的解 (包含:樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數, GA所挑選的特徵的編號)
    % 根據x所提供的RF超參數及訓練集(trainData)與對應的標籤(trainLabels), 
    % 使用x所提供的特徵項目建立迴歸RF模型, 再透過驗證集(validData)及其對應的
    % 標籤(validLabels)計算該RF模型的預測值的MSE,並作為ga的目標函數(fitnessFunction)
    fitnessFunction = @(x) RandomForestFitness(x, trainData, trainLabels, validData, validLabels,x(numParas+1:end));
    disp("演算模式: 採計GA挑選的 "+ num2str(numFeats) +"個特徵")
end



% 使用GA搜索最優設計變數值
[x,fval] = ga(fitnessFunction, numVariables, [], [], [], [], lb, ub, [],intcon,options);

%實際上, GA的運算結果並不是由上面的x與fval得到, 而是由ga中的'OutputFcn'所設定
% 的自訂gaoutputfunction.m, 由ga的每次疊代過程的資訊中所獲得


%[x]解答
%[fval]分數   the value of the fitness function at x
%[exitflag]結束運算的原因
%[output]輸出運算法的結構
%[population]最後一次疊代所有染色體的解答
%[scores]最後一次疊代所有染色體的分數
%Func-count調動目標函數的次數
%stall generations疊代沒有優化的次數



end