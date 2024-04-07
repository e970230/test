function fitness = RandomForestFitness(params, trainData, trainLabels, validData, validLabels, nov)
    % params: 要優化的隨機森林參數，例如決策樹數量、最大深度等
    % trainData: 訓練數據
    % trainLabels: 訓練數據對應標籤
    % validData: 驗證數據
    % validLabels: 驗證答案
    % 使用優化後的參數訓練隨機森林模型
    
    
%     if length(unique(nov(4:10))) < 7  % 7是因為有10項，確保第4到第10項不會有相同的數字
%         fitness = -Inf;  % 如果有重複，設定適性為負無窮
%         return;
%     end
    numTrees = params(1,1);
    maxDepth = params(1,2);
    minLeafSize = params(1,3);

    tr_trainData=trainData(:,nov(4:end));
    tr_vaildData=validData(:,nov(4:end));

    treeBaggerModel = TreeBagger(numTrees, tr_trainData, trainLabels, 'Method', 'regression', ...
        'MaxNumSplits', maxDepth, 'MinLeafSize', minLeafSize);
    
    % 在驗證集上進行預測
    predictions = predict(treeBaggerModel, tr_vaildData);
    
    % 計算預測性能指標
    MSE = mean((predictions - validLabels).^2);
    
    % 將目標函數設置為最小化的MSE
    fitness = MSE;
end