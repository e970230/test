%特徵篩選和整理


clc
clear
close all

%% 資料整理

input_data_0 = load("vibration_signal_feature_model_3_0_F16000.mat");
input_data_0 = input_data_0.Final_feature;

input_data_10 = load("vibration_signal_feature_model_3_10_F16000.mat");
input_data_10 = input_data_10.Final_feature;

input_current_data_0 = load("current_signal_feature_model_3_0_F16000.mat");
input_current_data_0 = input_current_data_0.feature;

input_current_data_10 = load("current_signal_feature_model_3_10_F16000.mat");
input_current_data_10 = input_current_data_10.feature;

output_data_0 = load("vibration_signal_label_model_3_0_F16000.mat");
output_data_0 = output_data_0.Fianl_labels;

output_data_10 = load("vibration_signal_label_model_3_10_F16000.mat");
output_data_10 = output_data_10.Fianl_labels;


input_data_total_0 = [input_data_0 input_current_data_0];
input_data_total_10 = [input_data_10 input_current_data_10];


%%


 feature_data = fisher_score([input_data_total_0;input_data_total_10],[output_data_0;output_data_10]);

 find_feature_num = 30;
% 找到前50個最大費雪分數的索引
[~, top_indices] = maxk(feature_data, find_feature_num);

rng(0)

random_numbers_to_0 = randperm(size(input_data_total_0,1));
random_numbers_to_10 = randperm(size(input_data_total_10,1));

%--------索引存放區---------------------------------------------------------

train_inx_0 = random_numbers_to_0(1:size(random_numbers_to_0,2)/2);
train_inx_10 = random_numbers_to_10(1:size(random_numbers_to_10,2)/2);


test_inx_0 =  random_numbers_to_0(size(random_numbers_to_0,2)/2+1:end);
test_inx_10 =  random_numbers_to_10(size(random_numbers_to_10,2)/2+1:end);

%--------------------------------------------------------------------------

train_input_data = [input_data_total_0(train_inx_0,top_indices);input_data_total_10(train_inx_10,top_indices)];
train_output_data = [output_data_0(train_inx_0);output_data_10(train_inx_10)];

test_input_data = [input_data_total_0(test_inx_0,top_indices);input_data_total_10(test_inx_10,top_indices)];
test_output_data = [output_data_0(test_inx_0);output_data_10(test_inx_10)];




%%



%讀取測試資料 (已由特徵排序方法篩選出前30名重要特徵)

FeatureData=train_input_data;    %特徵資料集; 維度=sample數*特徵數
FeatureData_label=train_output_data;  %特徵資料集對應的標籤資料; 維度=sample數*1


selection_method = 2;   %數據選取方法的設定
Split_quantity=5;       %利用k-fold平分數據之數量，請注意若使用方法1的選取方式時，此處請手動輸入data的標籤種類數量，意即若今天數據標籤有7種不同的標籤則此處請輸入7
ga_input=[20 300 0.7];   %GA參數設定, 依序為族群大小, 疊代次數上限, 交配率
numFeats=0;   %欲由GA挑選的特徵數量 (numFeats=0: 不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數)
lb_input=[20 5 5];     %欲優化的超參數的搜索範圍的下限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)
ub_input=[300 50 10];  %欲優化的超參數的搜索範圍的上限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)


tic

% 由於在ga_mix_tree_Fnc.m中已定義ga的'OutputFcn'為自訂的gaoutputfunction.m, 
% 其可將ga疊代過程的資訊儲存於Workspace(變數名稱: Population_answer), 故ga_mix_tree_Fnc.m毋須設定輸出即可獲得解答
[indices]=ga_mix_tree_Fnc(ga_input,numFeats,lb_input,ub_input,FeatureData,FeatureData_label,Split_quantity,'regression',selection_method);
%Population_answer為ga_mix_tree_Fnc運算完後自動整理出的歷史疊代資訊



%針對GA找出的參數解答, 將透過反覆重新建立RF模型來評估每個預測模型的預測值的重現性
Repeat_verification=30;  %重複建立RF的次數
[best_answer,best_score,verify,best_feature_inx] = ga_mix_tree_answer_Fnc(Population_answer,numFeats,Repeat_verification,FeatureData,FeatureData_label,Split_quantity,indices,'regression',selection_method);

% best_answer: 由GA找出的最佳的3項RF超參數; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
% best_score: 以GA全部疊代歷程中最佳(MSE最小)的染色體所代表的3項RF超參數所建立的RF預測模型, 於驗證集所得到的預測MSE值
% verify: 以最佳超參數及特徵重複多次建立RF, 對驗證集的預測值的fitness(MSE值)的平均值, 變異數, 標準差
% best_feature_inx: 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats); 若不須GA挑選特徵則回傳"[]"
toc


%% 驗證


final_score = RandomForestFitnessBasic(best_answer,test_input_data,test_output_data,Split_quantity,indices,'regression',1);


%建立RF模型(迴歸型)
treeBaggerModel = TreeBagger(best_answer(1), test_input_data, test_output_data, 'Method', 'regression', ...
    'MaxNumSplits', best_answer(2), 'MinLeafSize', best_answer(3));
    
predictions = predict(treeBaggerModel, test_input_data);      %以驗證集進行預測
fitness = mean((predictions - test_output_data).^2);     %計算預測值的MSE


disp("測試數據建立模型後進行kfold交叉驗證之MSE "+ num2str(final_score))

disp("測試數據建立模型後不進行kfold交叉驗證之MSE " + num2str(fitness))



