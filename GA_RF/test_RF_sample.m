% 此程式的目的為測試將資料分成多等分
% 並選擇不同的數據及來進行訓練和驗證

clc
clear
close all

% test_data=load('feature_dataset_top30.mat');
% test_data=test_data.feature_top30;
% test_data=test_data.dataset;

load("feature_dataset_heavy.mat")
X= feature_dataset(:,2:end);
Y= feature_dataset(:,1);

Split_quantity=5;
%% 測試新版的評分機制
indices = crossvalind('Kfold',Y,Split_quantity);
ans=RandomForestFitnessBasic([200 20 3],X,Y,Split_quantity,indices,'regression');


%% 拆分資料




indices = crossvalind('Kfold',test_data(:,1),Split_quantity);

for Sq=1:Split_quantity

    % 選取當前要使用作為測試的數據集編號
    exclude_label = Sq;
    
    % 找到選擇的測試數據編號以外的所有數據標籤
    include_indices = (indices ~= exclude_label);
    
    
    % 提取非排除类别的标签
    included_labels = indices(include_indices);
    
    
    
    ra_test_data=test_data(find(indices(:,:)==exclude_label),:);
    ra_test_data_label=test_data(find(indices(:,:)==exclude_label),1);
    ra_train_data=test_data(include_indices,:);
    ra_train_data_label=test_data(include_indices,1);



%     F(Sq)=RandomForestFitnessBasic([100 10 2],ra_train_data,ra_train_data_label,ra_test_data,ra_test_data_label);

end





