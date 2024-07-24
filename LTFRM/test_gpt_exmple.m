clc
clear
close all

init_data=load("feature_dataset_heavy.mat");
first_data=init_data.feature_dataset;
first_data_label = init_data.feature_name;


data=first_data(:,2:end);
labels=first_data(:,1);

%取得所有預壓力之標籤
label_sequence = unique(labels);


D = cell(length(label_sequence), 1);
for i = 1:length(label_sequence)
    D{i} = data(labels == label_sequence(i), :);
end


num_features = size(data, 2);
si = zeros(num_features, 1);

for i = 1:num_features  %多少特徵
    score = 0;
    for l = 1:(length(D) - 1)   %標籤種類數量減一
        
        mu_l = mean(D{l}(:, i));        %子資料集D裡面當下對應標籤所對應特徵之均值
        mu_l1 = mean(D{l + 1}(:, i));   %子資料集D裡面當下對應下一個標籤所對應特徵之均值
        
        % 計算 c 的判斷式
        if mu_l1 >= mu_l
            c = 1;
        else
            c = -1;
        end
        
        nDl1 = numel(D{l + 1}(:, i));   %子資料集D裡面當下對應標籤之資料點數
        nDl = numel(D{l}(:, i));        %子資料集D裡面當下對應下一個標籤之資料點數
        SDl1 = cov(D{l + 1}(:, i));     %子資料集D裡面當下對應標籤之共變異矩陣
        SDl = cov(D{l}(:, i));          %子資料集D裡面當下對應下一個標籤之共變異矩陣
        S_pool = (nDl1 / (nDl1 + nDl) * SDl1) + (nDl / (nDl1 + nDl) * SDl); %集合共變異矩陣公式
        
        % 特徵重要性分數計算
        score = score + (c * (mu_l1 - mu_l)^2) * inv(S_pool);
    end
    si(i) = abs(score);
end

%對si降冪排列得到特徵重要性排序
[~, feature_rank] = sort(si, 'descend');

answer=feature_rank(1:30);

for di=1:length(answer)
    disp("第" + num2str(di) + "名特徵" +first_data_label{answer(di)})
end

% figure;
% bar(an(1:30));
% title('Feature Importance Ranking');
% xlabel('Feature Index (Sorted)');
% ylabel('Importance Value');

