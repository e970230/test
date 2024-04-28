function fitness = RandomForestFitness(params, trainData, trainLabels, validData, validLabels, nov)
    % params: 要優化的隨機森林參數，例如決策樹數量、最大深度等
    % trainData: 訓練數據
    % trainLabels: 訓練數據對應標籤
    % validData: 驗證數據
    % validLabels: 驗證答案
    % nov: 所選擇的特徵
    % 使用優化後的參數訓練隨機森林模型
    numTrees = params(1,1);
    maxDepth = params(1,2);
    minLeafSize = params(1,3);
    %下面會從第四項開始是因為基因演算法在疊代時會將前面三個超參數設定一起綁在所選特徵數裡一起疊代
    %固要跳過前三項從第四項開始
    
    tr_trainData=trainData(:,nov);   %將選擇到的特徵當成訓練集
    tr_vaildData=validData(:,nov);   %將選擇到的特徵當成測試集


    treeBaggerModel = TreeBagger(numTrees, tr_trainData, trainLabels, 'Method', 'regression', ...
        'MaxNumSplits', maxDepth, 'MinLeafSize', minLeafSize);
    
    % 在驗證集上進行預測
    predictions = predict(treeBaggerModel, tr_vaildData);
    
    % 計算預測性能指標
    MSE = mean((predictions - validLabels).^2);
    % 將目標函數設置為最小化的MSE
    fitness = MSE;
end