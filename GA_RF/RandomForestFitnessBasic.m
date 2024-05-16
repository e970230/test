function [fitness] = RandomForestFitnessBasic(params, trainData, trainLabels, validData, validLabels)
% 根據所給定的RF超參數(樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)及訓練資料集(trainData)
% 與對應的標籤(trainLabels), 建立迴歸RF模型, 再透過驗證數據集(validData)及其對
% 應的標籤(validLabels)計算該RF模型的預測值的MSE(fitness)

% **在RF訓練與預測時, 皆採計訓練集與驗證集內的所有特徵項目**


% last modification: 2024/05/07


% input
%----------------------------------------------------
% params: RF的超參數設定值; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
% trainData: RF的訓練數據集; 維度=sample數*特徵數
% trainLabels: trainData對應的標籤(Label); 維度=sample數*1
% validData:  RF的驗證數據集; 維度=sample數*特徵數
% validLabels: validData對應的標籤(Label); 維度=sample數*1


% output
%----------------------------------------------------
% fitness: 迴歸型RF模型的預測值的MSE


numTrees = params(1);    %樹數目
maxNumSplits = params(2);    %每棵樹最大的分枝次數
minLeafSize = params(3); %葉節點最小樣本數

    
%建立RF模型(迴歸型)
treeBaggerModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'regression', ...
    'MaxNumSplits', maxNumSplits, 'MinLeafSize', minLeafSize);
    
predictions = predict(treeBaggerModel, validData);  %以驗證集進行預測
fitness = mean((predictions - validLabels).^2);  %計算預測值的MSE

end