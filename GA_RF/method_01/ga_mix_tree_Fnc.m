function[indices] = ga_mix_tree_Fnc(ga_input,numFeats,lb_input,ub_input,data,labels,Split_quantity,RF_mode,selection_method)

% 透過基因演算法(GA)對迴歸型(regression)隨機森林(RF)的超參數(樹數目, 每棵樹最大的分枝次數, 
% 葉節點最小樣本數)與特徵選擇進行優化; 並且使用k-fold的功能將原數據集(data)和其對應標籤(labels),
% 平分成Split_quantity份(原數據集和標籤中的所有類型都會被等分)；隨後取其中一份作為驗證數據集，其餘數據集為訓練資料集
% 搭配此三項最優超參數建立RF預測模型。
% 迴歸型RF目標函數: 計算該RF模型的預測準確度(MSE), 並以MSE作為GA疊代過程中的目標函數
% 分類型RF目標函數: 計算該RF模型的預測錯誤率，並以錯誤率作為GA疊代過程中的目標函數

% 2024/07/10:開始新增方法1之功能作為選擇，將某一特徵作為測試數據，其餘數據做為訓練數據

% input
%----------------------------------------------------
% ga_input: GA參數設定; 維度=1*3, 依序為 族群大小, 疊代次數上限, 交配率
% numFeats: 欲由GA挑選的特徵數量 
%           (numFeats=0: 不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數)
% lb_input: 欲優化的超參數的搜索範圍的下限; 維度=1*欲優化的超參數數目
% ub_input: 欲優化的超參數的搜索範圍的上限; 維度=1*欲優化的超參數數目
% data:     輸入RF的原數據集； 維度=樣本數*特徵數
% labels:   輸入RF的原數據集其對應標籤； 維度=樣本數*1
% Split_quantity: 將數據集拆分的數量，意即輸入4的話則代表拆成四份數據集，若selection_method設定為1則此行輸入請隨輸入數字
% indices:  由ga_mix_tree_Fnc.m對data進行交叉驗證前的數據拆分的編號集
%           由Split_quantity設定拆成多少份數據後，將每種不同的labels進行編號分出不同的數據集
%           假設將數據集分成四份，期對應的所有標籤都將分配1~4的編號並所有不同標籤都會等分成4份
% RF_mode:  可以選擇要使用分類型RF(classification)或是迴歸型RF(regression)，只需在使用時將對應的分類
%           輸入在此就行('regression'/'classification')
% selection_method: 設定如何選取測試資料和訓練資料的方法有1和2兩種
%                   1: 選取某一特定標籤做為測試資料，其餘做為訓練資料可以觀察模型在預測從未看過的標籤時模型的準度是否能如預期
%                      意即若今天有三種標籤則會選取其中一種標籤作為測試資料其餘兩種作為訓練資料，且每種標籤都會輪到作為測試資
%                      料，並讓三種狀況的預測值加起來平均後作為最終預測值
%                   2: kfold交叉驗證方式，並且是針對每種標籤都會選擇一定比例，而不存在於某種標籤被選到比較少的狀況

% output
%----------------------------------------------------
% indices: 由ga_mix_tree_Fnc.m對data進行交叉驗證前的數據拆分的編號集
%          由Split_quantity設定拆成多少份數據後，將每種不同的labels進行編號分出不同的數據集
%          假設將數據集分成四份，期對應的所有標籤都將分配1~4的編號並所有不同標籤都會等分成4份；維度=樣本數*1


% 使用子程式: RandomForestFitnessBasic.m, RandomForestFitness.m,
%            gaoutputfunction.m


%last modification: 2024/07/10



% 設定基因演算法(GA)的參數
PopulationSize=ga_input(1);     %族群大小 (染色體數目)
Generations=ga_input(2);        %疊代次數上限
CrossoverFraction=ga_input(3);  %交配率

out_regression=strcmp(RF_mode,'regression');            %進行迴歸型RF分類設定的字串比對
out_classification=strcmp(RF_mode,'classification');    %進行分類型RF分類設定的字串比對


options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'Generations',Generations,'CrossoverFraction', CrossoverFraction,'OutputFcn',@gaoutputfunction);
% ga的'OutputFcn'設定了自訂的gaoutputfunction.m, ga將於每次疊代時呼叫並執行此function



% 定義要優化的參數範圍
numParas=3;    %欲以GA優化的超參數數目 (因為是限定用於RF, 故超參數為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數等三種
numVariables = numParas+numFeats; % 要讓GA進行搜索的項目總數(GA的設計變數的數量) = 欲優化的3個超參數加上欲由GA挑選的特徵數目
intcon=1:numVariables;  %整數變數的數量


lb_sample=ones(1,numFeats); %欲由GA挑選的特徵的編號的最低數值
ub_sample=size(data,2)*ones(1,numFeats); %欲由GA挑選的特徵的編號的最高數值
lb = [lb_input lb_sample];  %每個設計變數的範圍(下限)
ub = [ub_input ub_sample];  %每個設計變數的範圍(上限)

if selection_method==1
    unm_unique=unique(labels);
    indices=zeros(length(labels),1);
    for i=1:length(unm_unique)
        indices(find(labels == unm_unique(i)),1)= i ;
    end
end

if selection_method==2
    indices = crossvalind('Kfold',labels,Split_quantity);   %將對應labels的所有標籤等分成Split_quantity份並給予編號

end

if out_regression==1   %字串相符時確認使用迴歸型RF         

    if numFeats==0  %不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數
    
        % x為ga.m所找到的解 (包含:樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)
        % 根據x所提供的RF超參數及訓練集(trainData)與對應的標籤(trainLabels), 
        % 建立迴歸RF模型, 再透過驗證集(validData)及其對應的標籤(validLabels)計算
        % 該RF模型的預測值的MSE,並作為ga的目標函數(fitnessFunction)
        fitnessFunction = @(x) RandomForestFitnessBasic(x,data,labels,Split_quantity,indices,'regression');
        disp("演算模式: 採計所有特徵")
    
    else
        
        % x為ga.m所找到的解 (包含:樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數, GA所挑選的特徵的編號)
        % 根據x所提供的RF超參數及訓練集(trainData)與對應的標籤(trainLabels), 
        % 使用x所提供的特徵項目建立迴歸RF模型, 再透過驗證集(validData)及其對應的
        % 標籤(validLabels)計算該RF模型的預測值的MSE,並作為ga的目標函數(fitnessFunction)
        fitnessFunction = @(x) RandomForestFitness(x,data,labels,Split_quantity,indices,x(numParas+1:end),'regression');
        disp("演算模式: 採計GA挑選的 "+ num2str(numFeats) +"個特徵")
    end
end

if out_classification==1    %字串相符時確認使用分類型RF 
    if numFeats==0  %不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數
    
        % x為ga.m所找到的解 (包含:樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)
        % 根據x所提供的RF超參數及訓練集(trainData)與對應的標籤(trainLabels), 
        % 建立迴歸RF模型, 再透過驗證集(validData)及其對應的標籤(validLabels)計算
        % 該RF模型的預測值的MSE,並作為ga的目標函數(fitnessFunction)
        fitnessFunction = @(x) RandomForestFitnessBasic(x,data,labels,Split_quantity,indices,'classification');
        disp("演算模式: 採計所有特徵")
    
    else
        
        % x為ga.m所找到的解 (包含:樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數, GA所挑選的特徵的編號)
        % 根據x所提供的RF超參數及訓練集(trainData)與對應的標籤(trainLabels), 
        % 使用x所提供的特徵項目建立迴歸RF模型, 再透過驗證集(validData)及其對應的
        % 標籤(validLabels)計算該RF模型的預測值的MSE,並作為ga的目標函數(fitnessFunction)
        fitnessFunction = @(x) RandomForestFitness(x,data,labels,Split_quantity,indices,x(numParas+1:end),'classification');
        disp("演算模式: 採計GA挑選的 "+ num2str(numFeats) +"個特徵")
    end

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