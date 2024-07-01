clc
clear
close all

%此為利用網路上別人寫的extratrees程式進行GA超參數優化之範例
%使用的資料為預壓力的資料集共計4858個特徵1298個樣本進行測試

load("feature_dataset_heavy.mat")

tic

data= feature_dataset(:,2:end);
label= feature_dataset(:,1);

indices = crossvalind('Kfold',label,5);



PopulationSize=10;     %族群大小 (染色體數目)
Generations=2;        %疊代次數上限
CrossoverFraction=0.7;  %交配率

options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'Generations',Generations,'CrossoverFraction', CrossoverFraction,'OutputFcn',@gaoutputfunction);
numVariables=3;
intcon=1:numVariables;  %整數變數的數量
lb = [10 5 2];  %每個設計變數的範圍(下限)
ub = [20 20 6];  %每個設計變數的範圍(上限)

fitnessFunction = @(x) ET_fit(data,label,x(1),x(2),x(3),indices);

[x,fval] = ga(fitnessFunction, numVariables, [], [], [], [], lb, ub, [],intcon,options);

toc

%遍歷各次疊代的過程
for pl=1:size(Population_answer,3)  %將最佳解(適性值、最佳的超參數配合)從相對應的疊代裡撈出來
    [~,answer_index(pl)]=min(Population_answer(:,end,pl));

    %將該次疊代時最佳(MSE最小)的染色體所代表的3項RF超參數與MSE值取出並另存至design_variables_history
    design_variables_history(pl,:)=Population_answer(answer_index(pl),1:end,pl);
end

[best_score,best_score_inx]=min(design_variables_history(:,end));  %在所有疊代過程中, 最佳(MSE最小)的染色體所代表的3項RF超參數與MSE值(best_score)
best_params=design_variables_history(best_score_inx,1:3);    %最佳的3項RF超參數
