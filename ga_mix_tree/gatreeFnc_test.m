clc
clear
close all
% 此為基因演算法控制回歸型(regression)隨機森林演算法超參數範例程式
% 此為ga_mix_tree_Fnc和ga_mix_tree_answer_Fnc的簡易使用說明下列都有將每個設定的變數做解說以下有幾點須注意
% 1.這兩種副程式都必須搭配3個副程式RandomForestFitness.m、RandomForestFitnessBasic.m、gaoutputfunction.m進行使用
%   1-1.RandomForestFitness.m為有選擇特徵的隨機森林適性值評分副程式
%   1-2.RandomForestFitnessBasic.m為無特別挑選特徵，已當前數據中所有特徵建立隨機森林適性值評分副程式
%   1-3.gaoutputfunction.m為抓取疊代資料副程式(當下疊代超參數組合跟當前的適性值分數，若有設定抓取特徵的話則一併存取)
% 2.確保訓練和驗證數據的矩陣是遵循 sample*feature
% 3.ga_mix_tree_Fnc找出的隨機森林超參數解目前就只有3個(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數)無法添加或減少尋找的超參數
% 4.若想在所有特徵中選取n個特徵，切記n不可以大過原本數據的特徵數量

tic
%% 示範案例

% data=load("feature_dataset_heavy.mat");
% first_data=data.feature_dataset;

feature_dataset=load("feature_dataset_top30.mat");
first_data=feature_dataset.feature_top30.dataset;

train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);

%全部資料集和標籤
trainData =train_data_all;          % 訓練數據 sample*feature
trainLabels =train_data_all_y;      % 訓練標籤 sample
validData = train_data_all;         % 驗證數據 sample*feature
validLabels = train_data_all_y;     % 驗證標籤 sample
ga_input=[10 100 0.7];             %基因演算法可供更改的超參數，切記不可空白3個參數都必須輸入(PopulationSize/Generations/CrossoverFraction)
sample_number=10;                   %選取的特徵數量(若要選取所有特徵則輸入0
lb_input=[50 5 5];                  %控制隨機森林3個參數的下限(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數)
ub_input=[200 50 10];               %控制隨機森林3個參數的上限(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數)
%由於內部自己寫的outputFnc有將歷代資訊建立在Workspace上(已Population_answer命名)，故這邊ga_mix_tree_Fnc沒有任何輸出即可自動得到解答
ga_mix_tree_Fnc(ga_input,sample_number,lb_input,ub_input,trainData, trainLabels, validData, validLabels);
%%
%Population_answer為ga_mix_tree_Fnc運算完後自動整理出的歷史疊代資訊
Repeat_verification=30; %針對已找出的參數解答，重複建立隨機森林的次數
[best_answer,best_score,verify,sample_index] = ga_mix_tree_answer_Fnc(Population_answer,sample_number,Repeat_verification,trainData, trainLabels, validData, validLabels);

%best_answer(最佳參數解)
%best_score(最佳適性值)
%verify已最佳參數解重複建立森林後所有適性值的(平均值/變異數/標準差)
%sample_index(選取的特徵，有排列過會由小到大排列)
toc