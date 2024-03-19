clc
clear
close all


%% 初始設立條件
feature_dataset=load("feature_dataset_heavy.mat");
first_data=feature_dataset.feature_dataset;
bit2int = @(bits) sum(bits .* 2.^(length(bits)-1:-1:0));
pool = parpool('local', 4); % 建立一個具有4個工作進程的本地平行運算池

rpm_120=first_data(find(first_data==120),:);
rpm_680=first_data(find(first_data==680),:);
rpm_890=first_data(find(first_data==890),:);
rpm_1100=first_data(find(first_data==1100),:);
rpm_1400=first_data(find(first_data==1400),:);
rpm_1500=first_data(find(first_data==1500),:);
rpm_2500=first_data(find(first_data==2500),:);

AA=[120 680 890 1100 1400 1500 2500];



rpm_680_size=size(rpm_680,1);
rpm_890_size=size(rpm_890,1);
rpm_1100_size=size(rpm_1100,1);
rpm_1400_size=size(rpm_1400,1);
rpm_1500_size=size(rpm_1500,1);
rpm_2500_size=size(rpm_2500,1);


%各轉速隨機數

rpm_120_rand=randperm(size(rpm_120,1));
rpm_680_rand=randperm(size(rpm_680,1));
rpm_890_rand=randperm(size(rpm_890,1));
rpm_1100_rand=randperm(size(rpm_1100,1));
rpm_1400_rand=randperm(size(rpm_1400,1));
rpm_1500_rand=randperm(size(rpm_1500,1));
rpm_2500_rand=randperm(size(rpm_2500,1));



randomly_selected_samples=100;
sampled_data=120;



% train_data=[rpm_120(rpm_120_rand(1:randomly_selected_samples),2:11);rpm_680(rpm_680_rand(1:randomly_selected_samples),2:11);rpm_890(rpm_890_rand(1:randomly_selected_samples),2:11);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),2:11);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),2:11);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),2:11);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),2:11)];
% train_data_y=[rpm_120(rpm_120_rand(1:randomly_selected_samples),1);rpm_680(rpm_680_rand(1:randomly_selected_samples),1);rpm_890(rpm_890_rand(1:randomly_selected_samples),1);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),1);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),1);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),1);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),1)];
% input_data=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),2:11);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),2:11);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),2:11);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),2:11);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),2:11);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),2:11);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),2:11)];
% input_data_answer=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),1);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),1);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),1);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),1);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),1);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),1);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),1)];
train_data_all=first_data(:,2:end);
train_data_all_y=first_data(:,1);

trainData =train_data_all; % 訓練數據
trainLabels =train_data_all_y; % 訓練標籤
validData = train_data_all; % 驗證數據
validLabels = train_data_all_y; % 驗證標籤

% 設置TreeBagger的選項，包括開啟平行運算
treeBaggerOptions = statset('UseParallel', true);


% for tre=1:10
tic

input_numTrees = 500;%森林中樹的數量
input_MinLeafSize=5;%葉節點最小樣本數
input_MaxNumSplits=10;%每顆樹最大的分割次數

treeBaggerModel = TreeBagger(input_numTrees, trainData, trainLabels,'Method','regression', ...
        'MinLeafSize',input_MinLeafSize,'MaxNumSplits',input_MaxNumSplits,'Options', treeBaggerOptions);
predictions = predict(treeBaggerModel, validData);
mse_answer=mse(predictions-validLabels);
mse_mean_answer=mean((predictions - validLabels).^2);
score=1./mse_answer;
toc
%%

% end
delete(gcp);
% true_answer=mse(input_data_answer)/size(input_data_answer,1);
ture_score=mse(validLabels);
ture=mse_answer;
disp(ture)
disp(score)
