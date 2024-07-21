clc
clear
close all

data=load("feature_dataset_heavy.mat");
first_data=data.feature_dataset;


train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);

label_of_tag=unique(train_data_all_y);  %標籤的種類數量

for L=1:length(label_of_tag)-1
    D{L} = train_data_all(find(train_data_all_y==label_of_tag(L)),:);
end



s_i={}; %特徵的重要性分數

for i=1:size(train_data_all,2)
    for L=1:length(label_of_tag)-1
        D_now=D{L};
        D_next=D{L+1};
        c=c_test(mean(D_next(:,i)),mean(D_now(:,i)));
        mum=mean(D_next(:,i))-mean(D_now(:,i));
        S=((numel(D_next)/(numel(D_next)+numel(D_now))*cov(D_next))+(numel(D_now)/(numel(D_next)+numel(D_now))*cov(D_now)));
        s_i{L,i}=abs(c*mum.^2*inv(S));
    end
end

% 計算共變異矩陣
covMatrix = cov(train_data_all);

% 顯示結果
% disp('數據矩陣:');
% disp(data);
% disp('共變異矩陣:');
% disp(covMatrix);




function [c] = c_test(l_plus_one,l_one)
    if l_plus_one >= l_one
        c=1;
    else
        c=-1;
    end
end