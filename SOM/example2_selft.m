clc
clear
close all

% 生成隨機數據
rng(1); % 設置隨機數種子，以確保結果可重複
numSamples = 1000; % 樣本數
numFeatures = 2; % 特徵數
random_data = randn(numFeatures, numSamples); % 生成隨機數據

% 對數據進行標準化
normalized_data = normalize(random_data, 2);

% 訓練 SOM 網路
dimensions = [10 10]; % SOM 網路維度
coverSteps = 100; % 訓練步數
initNeighbor = 3; % 初始鄰域大小
topologyFcn = 'hextop'; % 拓撲函數
distFcn = 'dist'; % 距離函數
net = selforgmap(dimensions, coverSteps, initNeighbor, topologyFcn, distFcn); % 創建 SOM 網路
SOM = train(net, normalized_data); % 訓練 SOM 網路

% 生成一個新的數據點來進行預測
new_data_point = randn(numFeatures, 1); % 生成一個隨機新數據點
normalized_new_data_point = normalize(new_data_point, 2); % 對新數據點進行標準化

% 找到新數據點的最佳匹配單元 (BMU)
isBMU = SOM(normalized_new_data_point); % 找到新數據點的 BMU
BMU_inx = find(isBMU == 1); % 找到 BMU 的索引
BMU_vector = net.IW{1}(:, BMU_inx); % 從 SOM 網路中取得 BMU 的權重向量

% 計算新數據點和其 BMU 的均方根誤差 (MQE)
MQE = sqrt(sum((BMU_vector-normalized_new_data_point).^2));

% 顯示結果
disp('Random data point:');
disp(new_data_point');
disp('Normalized random data point:');
disp(normalized_new_data_point');
disp('BMU vector:');
disp(BMU_vector');
disp(['Mean Quantization Error (MQE): ' num2str(MQE)]);