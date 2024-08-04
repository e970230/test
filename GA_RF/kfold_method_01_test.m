clc
clear
close all

data=load("feature_dataset_heavy.mat");
Data=data.feature_dataset;

FeatureData=Data(:,2:end);    %特徵資料集; 維度=sample數*特徵數
FeatureData_label=Data(:,1);  %特徵資料集對應的標籤資料; 維度=sample數*1

unm_unique=unique(FeatureData_label);
test_index=zeros(length(FeatureData_label),1);
for i=1:length(unm_unique)
    test_index(find(FeatureData_label == unm_unique(i)),1)= i ;
end

% for i=1:length(unm_unique)
%     trainData=FeatureData(find(FeatureData_label ~= unm_unique(i)),:);                %將數據集中未被選到做為驗證數據的數據存取出來(總樣本數減去被選為驗證數據樣本數*總特徵數)
%     trainLabels=FeatureData_label(find(FeatureData_label ~= unm_unique(i)),1);            %將標籤集中未被選到做為驗證標籤的標籤存取出來(總樣本數減去被選為驗證數據樣本數*1)
%     validData=FeatureData(find(FeatureData_label == unm_unique(i)),:);         %將數據集中被選到做為驗證數據的數據存取出來(被選為驗證數據樣本數*總特徵數)
%     validLabels=FeatureData_label(find(FeatureData_label == unm_unique(i)),1);     %將標籤集中被選到做為驗證標籤的標籤存取出來(被選為驗證數據樣本數*1)
% 
% end