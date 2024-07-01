% 以基因演算法(GA)決定迴歸型(regression)隨機森林(RF)的超參數(樹數目, 
% 每棵樹最大的分枝次數, 葉節點最小樣本數), 並挑選最適特徵

% 使用子程式: ga_mix_tree_Fnc.m, ga_mix_tree_answer_Fnc.m, gaoutputfunction.m,
%            RandomForestFitnessBasic.m, RandomForestFitness.m


% ga_mix_tree_Fnc.m
%--------------------------------------------------------------------------
% 透過基因演算法(GA)對迴歸型(regression)隨機森林(RF)的超參數(樹數目, 每棵樹最大的分枝次數, 
% 葉節點最小樣本數)與特徵選擇進行優化; 並且使用k-fold的功能將原數據集(data)和其對應標籤(labels),
% 平分成Split_quantity份(原數據集和標籤中的所有類型都會被等分)；隨後取其中一份作為驗證數據集，其餘數據集為訓練資料集
% 搭配此三項最優超參數建立RF預測模型。
% 迴歸型RF目標函數: 計算該RF模型的預測準確度(MSE), 並以MSE作為GA疊代過程中的目標函數
% 分類型RF目標函數: 計算該RF模型的預測錯誤率，並以錯誤率作為GA疊代過程中的目標函數


% gaoutputfunction.m
%--------------------------------------------------------------------------
% 在ga.m的'OutputFcn'設定了自訂的gaoutputfunction.m, ga將於每次疊代時呼叫並執行
% 此function. 實際上, 在本應用中, GA的運算結果並不是由ga.m的輸出x與fval得到, 
% 而是經由gaoutputfunction.m於ga的每次疊代過程中將相關資訊提取出所獲得


% ga_mix_tree_answer_Fnc.m
%--------------------------------------------------------------------------
%針對GA找出的參數解答, 將透過反覆重新建立RF模型來評估每個預測模型的預測值的重現性
%評判此種超參數設定答案是否會浮動過大



% RandomForestFitness.m
%--------------------------------------------------------------------------
% 根據所給定的RF超參數(樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)及資料集(data)
% 與對應的標籤(labels), 建立(迴歸RF模型/分類RF模型), 再準備(訓練數據和標籤)以及(驗證數據和標籤)會利用kfold交叉驗證的方式
% 將數據拆成Split_quantity份計算該(迴歸RF模型的預測值的MSE(fitness)/分類RF模型的預測值的錯誤率)
% 在RF訓練與預測時, 訓練集與驗證集內只採計所指定的特徵項目(nov)


% RandomForestFitnessBasic.m
%--------------------------------------------------------------------------

% 根據所給定的RF超參數(樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)及資料集(data)
% 與對應的標籤(labels), 建立(迴歸RF模型/分類RF模型), 再準備(訓練數據和標籤)以及(驗證數據和標籤)會利用kfold交叉驗證的方式
% 將數據拆成Split_quantity份計算該(迴歸RF模型的預測值的MSE(fitness)/分類RF模型的預測值的錯誤率)
% 在RF訓練與預測時, 皆採計訓練集與驗證集內的所有特徵項目



% last modification: 2024/06/10




clc
clear
close all


%% 示範案例1: 不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數

%讀取測試資料 (已由特徵排序方法篩選出前30名重要特徵)
feature_dataset=load("feature_dataset_top30.mat");
Data=feature_dataset.feature_top30.dataset;

FeatureData=Data(:,2:end);    %特徵資料集; 維度=sample數*特徵數
FeatureData_label=Data(:,1);  %特徵資料集對應的標籤資料; 維度=sample數*1



Split_quantity=4;       %利用k-fold平分數據之數量
ga_input=[10 300 0.7];   %GA參數設定, 依序為族群大小, 疊代次數上限, 交配率
numFeats=0;   %欲由GA挑選的特徵數量 (numFeats=0: 不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數)
lb_input=[50 5 5];     %欲優化的超參數的搜索範圍的下限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)
ub_input=[200 50 10];  %欲優化的超參數的搜索範圍的上限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)


tic

% 由於在ga_mix_tree_Fnc.m中已定義ga的'OutputFcn'為自訂的gaoutputfunction.m, 
% 其可將ga疊代過程的資訊儲存於Workspace(變數名稱: Population_answer), 故ga_mix_tree_Fnc.m毋須設定輸出即可獲得解答
[indices]=ga_mix_tree_Fnc(ga_input,numFeats,lb_input,ub_input,FeatureData,FeatureData_label,Split_quantity,'regression');
%Population_answer為ga_mix_tree_Fnc運算完後自動整理出的歷史疊代資訊



%針對GA找出的參數解答, 將透過反覆重新建立RF模型來評估每個預測模型的預測值的重現性
Repeat_verification=30;  %重複建立RF的次數
[best_answer,best_score,verify,best_feature_inx] = ga_mix_tree_answer_Fnc(Population_answer,numFeats,Repeat_verification,FeatureData,FeatureData_label,Split_quantity,indices,'regression');

% best_answer: 由GA找出的最佳的3項RF超參數; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
% best_score: 以GA全部疊代歷程中最佳(MSE最小)的染色體所代表的3項RF超參數所建立的RF預測模型, 於驗證集所得到的預測MSE值
% verify: 以最佳超參數及特徵重複多次建立RF, 對驗證集的預測值的fitness(MSE值)的平均值, 變異數, 標準差
% best_feature_inx: 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats); 若不須GA挑選特徵則回傳"[]"
toc



%% 示範案例2: 由GA挑選特徵

%讀取測試資料 (原始完整的特徵資料集)
data=load("feature_dataset_heavy.mat");
Data=data.feature_dataset;

FeatureData=Data(:,2:end);    %特徵資料集; 維度=sample數*特徵數
FeatureData_label=Data(:,1);  %特徵資料集對應的標籤資料; 維度=sample數*1


Split_quantity=5;       %利用k-fold平分數據之數量
ga_input=[10 300 0.7];   %GA參數設定, 依序為族群大小, 疊代次數上限, 交配率
numFeats=30;   %欲由GA挑選的特徵數量
lb_input=[50 5 5];     %欲優化的超參數的搜索範圍的下限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)
ub_input=[100 50 10];  %欲優化的超參數的搜索範圍的上限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)



tic

% 由於在ga_mix_tree_Fnc.m中已定義ga的'OutputFcn'為自訂的gaoutputfunction.m, 
% 其可將ga疊代過程的資訊儲存於Workspace(變數名稱: Population_answer), 故ga_mix_tree_Fnc.m毋須設定輸出即可獲得解答
[indices]=ga_mix_tree_Fnc(ga_input,numFeats,lb_input,ub_input,FeatureData,FeatureData_label,Split_quantity,'regression');
%Population_answer為ga_mix_tree_Fnc運算完後自動整理出的歷史疊代資訊


%針對GA找出的參數解答, 將透過反覆重新建立RF模型來評估每個預測模型的預測值的重現性
Repeat_verification=30;  %重複建立RF的次數
[best_answer,best_score,verify,best_feature_inx] = ga_mix_tree_answer_Fnc(Population_answer,numFeats,Repeat_verification,FeatureData,FeatureData_label,Split_quantity,indices,'regression');

% best_answer: 由GA找出的最佳的3項RF超參數; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
% best_score: 以GA全部疊代歷程中最佳(MSE最小)的染色體所代表的3項RF超參數所建立的RF預測模型, 於驗證集所得到的預測MSE值
% verify: 以最佳超參數及特徵重複多次建立RF, 對驗證集的預測值的fitness(MSE值)的平均值, 變異數, 標準差
% best_feature_inx: 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats); 若不須GA挑選特徵則回傳"[]"
toc