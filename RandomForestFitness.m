function fitness = RandomForestFitness(params, trainData, trainLabels, validData, validLabels)
    % params: 要優化的隨機森林參數，例如決策樹數量、最大深度等
    % trainData: 訓練數據
    % trainLabels: 訓練數據對應標籤
    % validData: 驗證數據
    % validLabels: 驗證答案
    % 使用優化後的參數訓練隨機森林模型
    numTrees = params(1,1);
    maxDepth = params(1,2);
    minLeafSize = params(1,3);
    
    treeBaggerModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'regression', ...
        'MaxNumSplits', maxDepth, 'MinLeafSize', minLeafSize);
    
    % 在驗證集上進行預測
    predictions = predict(treeBaggerModel, validData);
    
    % 計算預測性能指標
    MSE = mean((predictions - validLabels).^2);
    
    % 將目標函數設置為最小化的MSE
    fitness = MSE;
end