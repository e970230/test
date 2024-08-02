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

final_data=first_data(:,[1;feature_rank(1:30)+1]);

%%

clc
clear
close all

init_data=load("feature_dataset_heavy.mat");
first_data=init_data.feature_dataset;
first_data_label = init_data.feature_name;


data=first_data(:,2:end);
labels=first_data(:,1);

weight = [];
for i = 1 : size(data, 2) % 尋訪所有特徵i
    target_type = unique(labels); % 列出所有class
    i_score = 0;
    for j = 0 : (length(target_type)-2)
        idx1 = find(labels == target_type(length(target_type)-j));
        idx2 = find(labels == target_type(length(target_type)-j-1));
        u1 = mean(data(idx1, i), 1)';
        u2 = mean(data(idx2, i), 1)';
        i_score = i_score + sign(u1-u2)*MD(data(idx1, i), data(idx2, i));
    end
    weight = [weight; abs(i_score)];
end

[weight_score, feature_rank] = sort(weight, 'descend');

answer=feature_rank(1:30);

for di=1:length(answer)
    disp("第" + num2str(di) + "名特徵" +first_data_label{answer(di)})
end

final_data=first_data(:,[1;feature_rank(1:30)+1]);

function d = MD(x1, x2)
    n1 = size(x1, 1);
    n2 = size(x2, 1);
    u1 = mean(x1, 1)';
    u2 = mean(x2, 1)';
    X1 = x1';
    X2 = x2';
    for i = 1 : size(X1, 2)
        X1(:, i) = X1(:, i) - u1;
    end
    for i = 1 : size(X2, 2)
        X2(:, i) = X2(:, i) - u2;
    end
    s1 = X1*X1'/(n1-1);
    s2 = X2*X2'/(n2-1);
    S = (n1/(n1+n2)*s1)+(n2/(n1+n2)*s2);
    d = sqrt((u1-u2)'*inv(S)*(u1-u2));
end


% figure;
% bar(an(1:30));
% title('Feature Importance Ranking');
% xlabel('Feature Index (Sorted)');
% ylabel('Importance Value');

