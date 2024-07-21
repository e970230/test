clc
clear
close all

init_data=load("feature_dataset_heavy.mat");
first_data=init_data.feature_dataset;
first_data_label = init_data.feature_name;


data=first_data(:,2:end);
labels=first_data(:,1);



% 定義標籤的變化趨勢，例如 L3 -> L1 -> L2
label_sequence = unique(labels);
% label_sequence = [2500 1500 1400 1100 890 680 120];
% 步驟 2: 根據標籤變化趨勢重新劃分數據集
D = cell(length(label_sequence), 1);
for i = 1:length(label_sequence)
    D{i} = data(labels == label_sequence(i), :);
end

% 步驟 3: 計算每個特徵的重要性分數
num_features = size(data, 2);
si = zeros(num_features, 1);

for i = 1:num_features
    score = 0;
    for l = 1:(length(D) - 1)
        
        mu_l = mean(D{l}(:, i));
        mu_l1 = mean(D{l + 1}(:, i));
        
        % 計算 cl+1l
        if mu_l1 >= mu_l
            c = 1;
        else
            c = -1;
        end
        
        % 計算 Sl+1l
        nDl1 = numel(D{l + 1}(:, i));
        nDl = numel(D{l}(:, i));
        SDl1 = cov(D{l + 1}(:, i));
        SDl = cov(D{l}(:, i));
        S_pool = (nDl1 / (nDl1 + nDl) * SDl1) + (nDl / (nDl1 + nDl) * SDl);
        
        % 特徵重要性分數
        score = score + (c * (mu_l1 - mu_l)^2) * inv(S_pool);
    end
    si(i) = abs(score);
end

% 步驟 4: 對 si 降冪排列得到特徵重要性排序
[an, feature_rank] = sort(si, 'descend');

answer=feature_rank(1:30);



figure;
bar(an(1:30));
title('Feature Importance Ranking');
xlabel('Feature Index (Sorted)');
ylabel('Importance Value');

% 輸出特徵重要性分數及其排序
disp('特徵重要性分數:');
disp(si);
disp('特徵重要性排序:');
disp(feature_rank);

% disp(first_data_label{answer})