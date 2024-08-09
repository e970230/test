function[best_answer,best_score,verify,best_feature_inx] = ga_mix_tree_answer_Fnc(Population_answer,numFeats,Repeat_verification,data,labels,Split_quantity,indices,RF_mode,selection_method)
%針對GA找出的參數解答, 將透過反覆重新建立RF模型來評估每個預測模型的預測值的重現性
%評判此種超參數設定答案是否會浮動過大

% 2024/08/09:新增方法3作為選擇，可以開啟或關閉k-fold功能

% last modification: 2024/08/09


% input
%----------------------------------------------------
% Population_answer: 儲存了GA於各次疊代過程中, 各染色體的設計變數值(包含超參數與GA挑選的特徵的編號) 
%                    與fitness(MSE值);
%                    維度 = 族群大小*GA的設計變數的數量(=3項超參數+numFeats個特徵)*總疊代次數
% numFeats: 欲由GA挑選的特徵數量 
%           (numFeats=0: 不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數)
% Repeat_verification: 重複建立RF的次數 (重複驗證次數)
% data:     輸入RF的原數據集； 維度=樣本數*特徵數
% labels:   輸入RF的原數據集其對應標籤； 維度=樣本數*1
% Split_quantity: 將數據集拆分的數量，意即輸入4的話則代表拆成四份數據集
% indices:  由ga_mix_tree_Fnc.m對data進行交叉驗證前的數據拆分的編號集
%           由Split_quantity設定拆成多少份數據後，將每種不同的labels進行編號分出不同的數據集
%           假設將數據集分成四份，期對應的所有標籤都將分配1~4的編號並所有不同標籤都會等分成4份
% RF_mode:  可以選擇要使用分類型RF(classification)或是迴歸型RF(regression)，只需在使用時將對應的分類
%           輸入在此就行('regression'/'classification')
% selection_method: 設定如何選取測試資料和訓練資料的方法有(1、2、3)三種
%                   1: 選取某一特定標籤做為測試資料，其餘做為訓練資料可以觀察模型在預測從未看過的標籤時模型的準度是否能如預期
%                      意即若今天有三種標籤則會選取其中一種標籤作為測試資料其餘兩種作為訓練資料，且每種標籤都會輪到作為測試資
%                      料，並讓三種狀況的預測值加起來平均後作為最終預測值
%                   2: kfold交叉驗證方式，並且是針對每種標籤都會選擇一定比例，而不存在於某種標籤被選到比較少的狀況
%                   3: 不採用k-fold交叉驗證，代表訓練數據即驗證數據，使用在資料樣本數不多時的選擇


% output
%----------------------------------------------------
% best_answer: 由GA找出的最佳的3項RF超參數; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
% best_score: 以GA全部疊代歷程中最佳(MSE最小)的染色體所代表的3項RF超參數所建立的RF預測模型, 於驗證集所得到的預測MSE值
% verify: 以最佳超參數及特徵重複多次建立RF, 對驗證集的預測值的fitness(MSE值)的平均值, 變異數, 標準差
% best_feature_inx: 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats); 若不須GA挑選特徵則回傳"[]"

out_regression=strcmp(RF_mode,'regression');            %進行迴歸型RF分類設定的字串比對
out_classification=strcmp(RF_mode,'classification');    %進行分類型RF分類設定的字串比對

if selection_method==1

    k_fold_switch=1;                                        %k-fold功能開啟
    unm_unique=unique(labels);                              %找出標籤種類的數量
    Split_quantity = length(unm_unique);                    %根據標籤種類數量設定所需fold數

elseif selection_method==2
    
    k_fold_switch=1;                                        %k-fold功能開啟

elseif selection_method==3  %若使用選取方法3將k-flod功能關閉
    
    k_fold_switch=0;        %k-flod功能關閉
    
end

if numFeats==0  %不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數
    answer_index=[]; %儲存GA每次疊代時, 最佳(MSE最小)的染色體其位在矩陣Population_answer中的第幾個row; 維度=1*疊代次數
    design_variables_history=[]; %儲存GA每次疊代時, 最佳(MSE最小)的染色體所代表的3項RF超參數與MSE值; 維度=疊代次數*4 (4: 依序為3項RF超參數與MSE值)

    %遍歷各次疊代的過程
    for pl=1:size(Population_answer,3)  %將最佳解(適性值、最佳的超參數配合)從相對應的疊代裡撈出來
        [~,answer_index(pl)]=min(Population_answer(:,end,pl));

        %將該次疊代時最佳(MSE最小)的染色體所代表的3項RF超參數與MSE值取出並另存至design_variables_history
        design_variables_history(pl,:)=Population_answer(answer_index(pl),1:end,pl);
    end

    [best_score,best_score_inx]=min(design_variables_history(:,end));  %在所有疊代過程中, 最佳(MSE最小)的染色體所代表的3項RF超參數與MSE值(best_score)
    best_params=design_variables_history(best_score_inx,1:3);    %最佳的3項RF超參數
    % design_variables_history: 把每次疊代時, 該次疊代中的最優染色體挑出, 並依照疊代順序儲存於design_variables_history中
    % design_variables_history(:,1:3): 紀錄各次疊代中的最優3項RF超參數
    % design_variables_history(:,end): 紀錄各次疊代中的最優的MSE值
    % best_score_inx: 在design_variables_history中, 所有疊代中的最優解位於第best_score_inx次疊代
    
    if out_regression==1    %字串相符時確認使用迴歸型RF 
    
        for i=1:Repeat_verification  %重複建立RF以評估GA找到的最佳超參數所建立模型的預測重現性
    
            best_numTrees = best_params(1);  %最佳超參數: 樹數目
            best_MaxNumSplits= best_params(2);  %最佳超參數: 每棵樹最大的分枝次數
            best_MinLeafSize= best_params(3);  %最佳超參數: 葉節點最小樣本數
            %以完全一樣的超參數重複多次建立RF, 可得到不同的預測MSE值
            fitness(i) = RandomForestFitnessBasic([best_numTrees,best_MaxNumSplits,best_MinLeafSize],data,labels,Split_quantity,indices,'regression',k_fold_switch);
            disp(["第" num2str(i) "次適性值: " num2str(fitness(i))])
        end
    end
    if out_classification==1    %字串相符時確認使用分類型RF 
        for i=1:Repeat_verification  %重複建立RF以評估GA找到的最佳超參數所建立模型的預測重現性
    
            best_numTrees = best_params(1);  %最佳超參數: 樹數目
            best_MaxNumSplits= best_params(2);  %最佳超參數: 每棵樹最大的分枝次數
            best_MinLeafSize= best_params(3);  %最佳超參數: 葉節點最小樣本數
            %以完全一樣的超參數重複多次建立RF, 可得到不同的預測MSE值
            fitness(i) = RandomForestFitnessBasic([best_numTrees,best_MaxNumSplits,best_MinLeafSize],data,labels,Split_quantity,indices,'classification',k_fold_switch);
            disp(["第" num2str(i) "次錯誤率: " num2str(fitness(i))])
        end
    end

    
    %再利用平均數、變異數、標準差等方法來評判此種超參數設定答案是否會浮動過大
    %進而避免導致出現人品爆棚的答案，誤以為是最佳解
    mean_fitness=mean(fitness,2);
    var_fitness=var(fitness,0,2);
    std_fitness=std(fitness,0,2);
    disp('樹數目')
    disp(best_numTrees);
    disp('每棵樹最大的分枝次數')
    disp(best_MaxNumSplits); 
    disp('葉節點最小樣本數')
    disp(best_MinLeafSize);
    
    if out_regression==1
        disp('歷代最佳分數(MSE值)')
        disp(best_score);
    elseif out_classification==1
        disp('歷代最佳分數(錯誤率)')
        disp(best_score);
    end
    
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得平均值" + num2str(mean_fitness))
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得變異數" + num2str(var_fitness))
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得標準差" + num2str(std_fitness))
    best_answer=best_params(1:end);  % 由GA找出的最佳的3項RF超參數
    verify=[mean_fitness var_fitness std_fitness];
    best_feature_inx=[];  % 由GA找出的最佳特徵的編號; 因不須GA挑選特徵故回傳"[]"


    if out_regression==1
        figure(2)
        plot(1:size(design_variables_history,1),design_variables_history(:,end));
        title('適性值疊代過程')
        xlabel('疊代次數')
        ylabel('每次疊代最佳適性值')
    elseif out_classification==1
        figure(2)
        plot(1:size(design_variables_history,1),design_variables_history(:,end));
        title('錯誤率疊代過程')
        xlabel('疊代次數')
        ylabel('每次疊代最佳錯誤率')
    end


else

    answer_index=[]; %儲存GA每次疊代時, 最佳(MSE最小)的染色體其位在矩陣Population_answer中的第幾個row; 維度=1*疊代次數
    design_variables_history=[]; %儲存GA每次疊代時, 最佳(MSE最小)的染色體所代表的3項RF超參數與MSE值; 維度=疊代次數*4 (4: 依序為3項RF超參數與MSE值)
    %遍歷各次疊代的過程
    for pl=1:size(Population_answer,3)  %將最佳解(適性值、最佳的超參數配合)從相對應的疊代裡撈出來
        [~,answer_index(pl)]=min(Population_answer(:,end,pl));

        %將該次疊代時最佳(MSE最小)的染色體所代表的3項RF超參數與MSE值取出並另存至design_variables_history
        design_variables_history(pl,:)=Population_answer(answer_index(pl),1:end,pl);
    end
    
    [best_score,best_score_inx]=min(design_variables_history(:,end));  %在所有疊代過程中, 最佳(MSE最小)的染色體所代表的3項RF超參數與MSE值(best_score)
    best_params=design_variables_history(best_score_inx,1:3);    %最佳的3項RF超參數
    % design_variables_history: 把每次疊代時, 該次疊代中的最優染色體挑出, 並依照疊代順序儲存於design_variables_history中
    % design_variables_history(:,1:3): 紀錄各次疊代中的最優3項RF超參數
    % design_variables_history(:,end): 紀錄各次疊代中的最優的MSE值
    % best_score_inx: 在design_variables_history中, 所有疊代中的最優解位於第best_score_inx次疊代


    % Population_answer的維度 = 族群大小*GA的設計變數的數量(=3項超參數+numFeats個特徵)*總疊代次數
    % 從中找到具有最優MSE值的染色體
    best_itaration=Population_answer(:,:,best_score_inx);   %由Population_answer中取出具有最優解的該次疊代; 維度=族群大小*GA的設計變數的數量(=3項超參數+numFeats個特徵)
    [~,best_chro_inx]=min(best_itaration(:,end));      %從中找到具有最優MSE值的染色體在族群中的索引編號
    nov=best_itaration(best_chro_inx,4:end-1);   %由GA挑選的特徵的編號; 維度=1*numFeats
    
    if out_regression==1    %字串相符時確認使用迴歸型RF
    
        for i=1:Repeat_verification  %重複建立RF以評估GA找到的最佳超參數所建立模型的預測重現性
            best_numTrees = best_params(1);    %最佳超參數: 樹數目
            best_MaxNumSplits= best_params(2);  %最佳超參數: 每棵樹最大的分枝次數
            best_MinLeafSize= best_params(3);  %最佳超參數: 葉節點最小樣本數
            %將完全一樣的條件建立多次的森林得到浮動的適性值
            fitness(i) = RandomForestFitness([best_numTrees,best_MaxNumSplits,best_MinLeafSize],data,labels,Split_quantity,indices, nov,'regression',k_fold_switch);
            disp(["第" num2str(i) "次適性值: " num2str(fitness(i))])
        end
    end
    if out_classification==1    %字串相符時確認使用分類型RF
        for i=1:Repeat_verification  %重複建立RF以評估GA找到的最佳超參數所建立模型的預測重現性
            best_numTrees = best_params(1);    %最佳超參數: 樹數目
            best_MaxNumSplits= best_params(2);  %最佳超參數: 每棵樹最大的分枝次數
            best_MinLeafSize= best_params(3);  %最佳超參數: 葉節點最小樣本數
            %將完全一樣的條件建立多次的森林得到浮動的適性值
            fitness(i) = RandomForestFitness([best_numTrees,best_MaxNumSplits,best_MinLeafSize],data,labels,Split_quantity,indices, nov,'classification',k_fold_switch);
            disp(["第" num2str(i) "次錯誤率: " num2str(fitness(i))])
        end
    end


    %再利用平均數、變異數、標準差等方法來評判此種超參數設定答案是否會浮動過大
    %進而避免導致出現人品爆棚的答案，誤以為是最佳解
    mean_fitness=mean(fitness,2);
    var_fitness=var(fitness,0,2);
    std_fitness=std(fitness,0,2);
    disp('樹數目')
    disp(best_numTrees);
    disp('每棵樹最大的分枝次數')
    disp(best_MaxNumSplits); 
    disp('葉節點最小樣本數')
    disp(best_MinLeafSize);
    
    if out_regression==1
        disp('歷代最佳分數(MSE值)')
        disp(best_score);
    elseif out_classification==1
        disp('歷代最佳分數(錯誤率)')
        disp(best_score);
    end
    
    disp('選擇的特徵')
    disp(sort(nov));
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得平均值" + num2str(mean_fitness))
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得變異數" + num2str(var_fitness))
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得標準差" + num2str(std_fitness))
    best_answer=best_params(1:end);  % 由GA找出的最佳的3項RF超參數
    verify=[mean_fitness var_fitness std_fitness];
    best_feature_inx=sort(nov);  % 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats)

    if out_regression==1
        figure(2)
        plot(1:size(design_variables_history,1),design_variables_history(:,end));
        title('適性值疊代過程')
        xlabel('疊代次數')
        ylabel('每次疊代最佳適性值')
    elseif out_classification==1
        figure(2)
        plot(1:size(design_variables_history,1),design_variables_history(:,end));
        title('錯誤率疊代過程')
        xlabel('疊代次數')
        ylabel('每次疊代最佳錯誤率')
    end

end


end