clc
clear
close all

%GA_RF的有趣測試，主要是想看以選擇TOP30特徵跟透過基因演算法選擇特徵時MSE值得收斂狀況如何

%新增觀察每次最佳的超參數答案組合

test_time = 3:1:15;

%-------------------------原始數據透過基因演算法抽取特徵---------------------

selection_method = 3;   %數據選取方法的設定
Split_quantity=7;       %利用k-fold平分數據之數量，請注意若使用方法1的選取方式時，此處請手動輸入data的標籤種類數量，意即若今天數據標籤有7種不同的標籤則此處請輸入7
numFeats=30;   %欲由GA挑選的特徵數量
lb_input=[50 5 5];     %欲優化的超參數的搜索範圍的下限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)
ub_input=[200 50 10];  %欲優化的超參數的搜索範圍的上限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)

%讀取測試資料 (原始完整的特徵資料集)
data=load("feature_dataset_heavy.mat");
Data=data.feature_dataset;
FeatureData=Data(:,2:end);    %特徵資料集; 維度=sample數*特徵數
FeatureData_label=Data(:,1);  %特徵資料集對應的標籤資料; 維度=sample數*1
time_records_1 = [];
answer_1 = [];
%--------------------------------------------------------------------------



for i_test = test_time

    ga_input=[10 i_test*10 0.7];   %GA參數設定, 依序為族群大小, 疊代次數上限, 交配率
    
    tic
    
    % 由於在ga_mix_tree_Fnc.m中已定義ga的'OutputFcn'為自訂的gaoutputfunction.m, 
    % 其可將ga疊代過程的資訊儲存於Workspace(變數名稱: Population_answer), 故ga_mix_tree_Fnc.m毋須設定輸出即可獲得解答
    [indices]=ga_mix_tree_Fnc(ga_input,numFeats,lb_input,ub_input,FeatureData,FeatureData_label,Split_quantity,'regression',selection_method);
    %Population_answer為ga_mix_tree_Fnc運算完後自動整理出的歷史疊代資訊
    
    
    %針對GA找出的參數解答, 將透過反覆重新建立RF模型來評估每個預測模型的預測值的重現性
    Repeat_verification=3;  %重複建立RF的次數
    [best_answer,best_score,verify,best_feature_inx] = ga_mix_tree_answer_Fnc(Population_answer,numFeats,Repeat_verification,FeatureData,FeatureData_label,Split_quantity,indices,'regression',selection_method);
    
    % best_answer: 由GA找出的最佳的3項RF超參數; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
    % best_score: 以GA全部疊代歷程中最佳(MSE最小)的染色體所代表的3項RF超參數所建立的RF預測模型, 於驗證集所得到的預測MSE值
    % verify: 以最佳超參數及特徵重複多次建立RF, 對驗證集的預測值的fitness(MSE值)的平均值, 變異數, 標準差
    % best_feature_inx: 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats); 若不須GA挑選特徵則回傳"[]"
    
    elapsed_time = toc;
    time_records_1 = [time_records_1; elapsed_time]; % 將每次的時間記錄到矩陣中
    answer_1(i_test,:) = [best_score best_answer best_feature_inx];

end

%-------------------------TOP30數據----------------------------------------

selection_method = 3;   %數據選取方法的設定
Split_quantity=7;       %利用k-fold平分數據之數量，請注意若使用方法1的選取方式時，此處請手動輸入data的標籤種類數量，意即若今天數據標籤有7種不同的標籤則此處請輸入7
numFeats=0;   %欲由GA挑選的特徵數量
lb_input=[50 5 5];     %欲優化的超參數的搜索範圍的下限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)
ub_input=[200 50 10];  %欲優化的超參數的搜索範圍的上限 (依序為RF的樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數)

%讀取測試資料 (原始完整的特徵資料集)
feature_dataset=load("feature_dataset_top30.mat");
Data=feature_dataset.feature_top30.dataset;
FeatureData=Data(:,2:end);    %特徵資料集; 維度=sample數*特徵數
FeatureData_label=Data(:,1);  %特徵資料集對應的標籤資料; 維度=sample數*1
time_records_2 = [];
answer_2 = [];
%--------------------------------------------------------------------------



for i_test = test_time

    ga_input=[10 i_test*10 0.7];   %GA參數設定, 依序為族群大小, 疊代次數上限, 交配率
    
    tic
    
    % 由於在ga_mix_tree_Fnc.m中已定義ga的'OutputFcn'為自訂的gaoutputfunction.m, 
    % 其可將ga疊代過程的資訊儲存於Workspace(變數名稱: Population_answer), 故ga_mix_tree_Fnc.m毋須設定輸出即可獲得解答
    [indices]=ga_mix_tree_Fnc(ga_input,numFeats,lb_input,ub_input,FeatureData,FeatureData_label,Split_quantity,'regression',selection_method);
    %Population_answer為ga_mix_tree_Fnc運算完後自動整理出的歷史疊代資訊
    
    
    %針對GA找出的參數解答, 將透過反覆重新建立RF模型來評估每個預測模型的預測值的重現性
    Repeat_verification=3;  %重複建立RF的次數
    [best_answer,best_score,verify,best_feature_inx] = ga_mix_tree_answer_Fnc(Population_answer,numFeats,Repeat_verification,FeatureData,FeatureData_label,Split_quantity,indices,'regression',selection_method);
    
    % best_answer: 由GA找出的最佳的3項RF超參數; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
    % best_score: 以GA全部疊代歷程中最佳(MSE最小)的染色體所代表的3項RF超參數所建立的RF預測模型, 於驗證集所得到的預測MSE值
    % verify: 以最佳超參數及特徵重複多次建立RF, 對驗證集的預測值的fitness(MSE值)的平均值, 變異數, 標準差
    % best_feature_inx: 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats); 若不須GA挑選特徵則回傳"[]"
    
    elapsed_time = toc;
    time_records_2 = [time_records_2; elapsed_time]; % 將每次的時間記錄到矩陣中
    answer_2(i_test,:) = [best_score best_answer best_feature_inx];

end




figure(3)
hold on
plot(test_time*10,answer_1(test_time,1),'Color','r')%畫出分數隨著疊代次數的走勢圖
plot(test_time*10,answer_2(test_time,1),'Color','b')%畫出分數隨著疊代次數的走勢圖
hold off
title('最佳適性值走勢圖')
legend('基因演算法尋找特徵','LTFRM找出之TOP30特徵')
xlabel('當次總疊代次數')
ylabel('當次完整疊代最佳MSE值')

figure(4)
hold on
plot(test_time*10,time_records_1,'Color','r')
plot(test_time*10,time_records_2,'Color','b')
hold off
title('花費時間走勢圖')
legend('基因演算法尋找特徵','LTFRM找出之TOP30特徵')
xlabel('當次總疊代次數')
ylabel('當次完整疊代花費時間')
