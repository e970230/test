function [fitness] = RandomForestFitnessBasic(params,data,labels,Split_quantity,indices,RF_mode)
% 根據所給定的RF超參數(樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)及資料集(data)
% 與對應的標籤(labels), 建立(迴歸RF模型/分類RF模型), 再準備(訓練數據和標籤)以及(驗證數據和標籤)會利用kfold交叉驗證的方式
% 將數據拆成Split_quantity份計算該(迴歸RF模型的預測值的MSE(fitness)/分類RF模型的預測值的錯誤率)

% 關於kfold交叉驗證方式說明如下:
% 假使將數據拆成4份意即對應原數據下的每種對應標籤平分成4份(每種標籤都會平分)
% 接下來從這四份數據裡挑選其中一份數據做為驗證數據和對應標籤，其餘三分作為訓練數據和對應標籤
% 且四份數據都會被選中做為驗證數據和對應標籤依此類推，也就代表會有四個預測值的MSE(fitness)
% 最後將四個預測值取平均作為最後預測值

% **在RF訓練與預測時, 皆採計訓練集與驗證集內的所有特徵項目**


% last modification: 2024/06/10


% input
%----------------------------------------------------
% params: RF的超參數設定值; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
% data: 輸入RF的原數據集； 維度=樣本數*特徵數
% labels: 輸入RF的原數據集其對應標籤； 維度=樣本數*1
% Split_quantity: 將數據集拆分的數量，意即輸入4的話則代表拆成四份數據集

% indices: 由ga_mix_tree_Fnc.m對data進行交叉驗證前的數據拆分的編號集
%          由Split_quantity設定拆成多少份數據後，將每種不同的labels進行編號分出不同的數據集
%          假設將數據集分成四份，期對應的所有標籤都將分配1~4的編號並所有不同標籤都會等分成4份
% RF_mode: 可以選擇要使用分類型RF(classification)或是迴歸型RF(regression)，只需在使用時將對應的分類
%          輸入在此就行('regression'/'classification')

% output
%----------------------------------------------------
% fitness: 迴歸型RF模型的預測值的MSE/分類型RF模型的預測值為預測答案之錯誤率
% 

out_regression=strcmp(RF_mode,'regression');            %進行迴歸型RF分類設定的字串比對
out_classification=strcmp(RF_mode,'classification');    %進行分類型RF分類設定的字串比對

numTrees = params(1);    %樹數目
maxNumSplits = params(2);    %每棵樹最大的分枝次數
minLeafSize = params(3); %葉節點最小樣本數

if out_regression==1    %字串相符時確認使用迴歸型RF  

    for Sq=1:Split_quantity

        % 找到選擇的測試數據編號以外的所有數據標籤
        include_indices = (indices ~= Sq);
        
        
        trainData=data(include_indices,:);                %將數據集中未被選到做為驗證數據的數據存取出來(總樣本數減去被選為驗證數據樣本數*總特徵數)
        trainLabels=labels(include_indices,1);            %將標籤集中未被選到做為驗證標籤的標籤存取出來(總樣本數減去被選為驗證數據樣本數*1)
        validData=data(find(indices(:,:)==Sq),:);         %將數據集中被選到做為驗證數據的數據存取出來(被選為驗證數據樣本數*總特徵數)
        validLabels=labels(find(indices(:,:)==Sq),1);     %將標籤集中被選到做為驗證標籤的標籤存取出來(被選為驗證數據樣本數*1)
        
        %建立RF模型(迴歸型)
        treeBaggerModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'regression', ...
            'MaxNumSplits', maxNumSplits, 'MinLeafSize', minLeafSize);
            
        predictions = predict(treeBaggerModel, validData);  %以驗證集進行預測
        fitness(Sq) = mean((predictions - validLabels).^2);  %計算預測值的MSE

    end
end
if out_classification==1    %字串相符時確認使用分類型RF 
    for Sq=1:Split_quantity

    
        % 找到選擇的測試數據編號以外的所有數據標籤
        include_indices = (indices ~= Sq);
        
        
        trainData=data(include_indices,:);                %將數據集中未被選到做為驗證數據的數據存取出來(總樣本數減去被選為驗證數據樣本數*總特徵數)
        trainLabels=labels(include_indices,1);            %將標籤集中未被選到做為驗證標籤的標籤存取出來(總樣本數減去被選為驗證數據樣本數*1)
        validData=data(find(indices(:,:)==Sq),:);         %將數據集中被選到做為驗證數據的數據存取出來(被選為驗證數據樣本數*總特徵數)
        validLabels=labels(find(indices(:,:)==Sq),1);     %將標籤集中被選到做為驗證標籤的標籤存取出來(被選為驗證數據樣本數*1)
        
        %建立RF模型(分類型)
        treeBaggerModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'classification', ...
            'MaxNumSplits', maxNumSplits, 'MinLeafSize', minLeafSize);
            
        predictions = abs(str2double(predict(treeBaggerModel, validData))-validLabels);  %以驗證集進行預測
        error_time=(length(find(predictions==~0))); %計算預測答案和驗證標籤的錯誤次數
        
        fitness(Sq) = error_time/size(validLabels,1);  %計算錯誤次數占整體答案的百分比(若預測答案完全預測正確則此項為0)

    end

end


fitness=mean(fitness);  %將kfold交叉驗證完後所得的所有分數取平均做為最終分數

end