function[best_answer,best_score,verify,best_feature_inx] = ga_mix_tree_answer_Fnc(Population_answer,numFeats,Repeat_verification,data,labels,Split_quantity,indices,RF_mode)
%針對GA找出的參數解答, 將透過反覆重新建立RF模型來評估每個預測模型的預測值的重現性
%評判此種超參數設定答案是否會浮動過大


% last modification: 2024/05/07


% input
%----------------------------------------------------
% Population_answer: 儲存了GA於各次疊代過程中, 各染色體的設計變數值(包含超參數與GA挑選的特徵的編號) 
%                    與fitness(MSE值);
%                    維度 = 族群大小*GA的設計變數的數量(=3項超參數+numFeats個特徵)*總疊代次數
% numFeats: 欲由GA挑選的特徵數量 
%           (numFeats=0: 不經由GA進行特徵選擇, 即以所有特徵進行建模以便優化超參數)
% Repeat_verification: 重複建立RF的次數 (重複驗證次數)
% trainData: RF的訓練數據集; 維度=sample數*特徵數
% trainLabels: trainData對應的標籤(Label); 維度=sample數*1
% validData:  RF的驗證數據集; 維度=sample數*特徵數
% validLabels: validData對應的標籤(Label); 維度=sample數*1


% output
%----------------------------------------------------
% best_answer: 由GA找出的最佳的3項RF超參數; 維度=1*3, 依序為樹數目, 每棵樹最大的分枝次數, 葉節點最小樣本數
% best_score: 以GA全部疊代歷程中最佳(MSE最小)的染色體所代表的3項RF超參數所建立的RF預測模型, 於驗證集所得到的預測MSE值
% verify: 以最佳超參數及特徵重複多次建立RF, 對驗證集的預測值的fitness(MSE值)的平均值, 變異數, 標準差
% best_feature_inx: 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats); 若不須GA挑選特徵則回傳"[]"

out_regression=strcmp(RF_mode,'regression');
out_classification=strcmp(RF_mode,'classification');

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
    
    if out_regression==1
    
        for i=1:Repeat_verification  %重複建立RF以評估GA找到的最佳超參數所建立模型的預測重現性
    
            best_numTrees = best_params(1);  %最佳超參數: 樹數目
            best_MaxNumSplits= best_params(2);  %最佳超參數: 每棵樹最大的分枝次數
            best_MinLeafSize= best_params(3);  %最佳超參數: 葉節點最小樣本數
            %以完全一樣的超參數重複多次建立RF, 可得到不同的預測MSE值
            fitness(i) = RandomForestFitnessBasic([best_numTrees,best_MaxNumSplits,best_MinLeafSize],data,labels,Split_quantity,indices,'regression');
            disp(["第" num2str(i) "次適性值: " num2str(fitness(i))])
        end
    end
    if out_classification==1
        for i=1:Repeat_verification  %重複建立RF以評估GA找到的最佳超參數所建立模型的預測重現性
    
            best_numTrees = best_params(1);  %最佳超參數: 樹數目
            best_MaxNumSplits= best_params(2);  %最佳超參數: 每棵樹最大的分枝次數
            best_MinLeafSize= best_params(3);  %最佳超參數: 葉節點最小樣本數
            %以完全一樣的超參數重複多次建立RF, 可得到不同的預測MSE值
            fitness(i) = RandomForestFitnessBasic([best_numTrees,best_MaxNumSplits,best_MinLeafSize],data,labels,Split_quantity,indices,'classification');
            disp(["第" num2str(i) "次適性值: " num2str(fitness(i))])
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
    disp('歷代最佳分數')
    disp(best_score);
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得平均值" + num2str(mean_fitness))
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得變異數" + num2str(var_fitness))
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得標準差" + num2str(std_fitness))
    best_answer=best_params(1:end);  % 由GA找出的最佳的3項RF超參數
    verify=[mean_fitness var_fitness std_fitness];
    best_feature_inx=[];  % 由GA找出的最佳特徵的編號; 因不須GA挑選特徵故回傳"[]"


    figure(2)
    plot(1:size(design_variables_history,1),design_variables_history(:,end));
    title('適性值疊代過程')
    xlabel('疊代次數')
    ylabel('每次疊代最佳適性值')


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
    
    if out_regression==1
    
        for i=1:Repeat_verification  %重複建立RF以評估GA找到的最佳超參數所建立模型的預測重現性
            best_numTrees = best_params(1);    %最佳超參數: 樹數目
            best_MaxNumSplits= best_params(2);  %最佳超參數: 每棵樹最大的分枝次數
            best_MinLeafSize= best_params(3);  %最佳超參數: 葉節點最小樣本數
            %將完全一樣的條件建立多次的森林得到浮動的適性值
            fitness(i) = RandomForestFitness([best_numTrees,best_MaxNumSplits,best_MinLeafSize],data,labels,Split_quantity,indices, nov,'regression');
            disp(["第" num2str(i) "次適性值: " num2str(fitness(i))])
        end
    end
    if out_classification==1
        for i=1:Repeat_verification  %重複建立RF以評估GA找到的最佳超參數所建立模型的預測重現性
            best_numTrees = best_params(1);    %最佳超參數: 樹數目
            best_MaxNumSplits= best_params(2);  %最佳超參數: 每棵樹最大的分枝次數
            best_MinLeafSize= best_params(3);  %最佳超參數: 葉節點最小樣本數
            %將完全一樣的條件建立多次的森林得到浮動的適性值
            fitness(i) = RandomForestFitness([best_numTrees,best_MaxNumSplits,best_MinLeafSize],data,labels,Split_quantity,indices, nov,'classification');
            disp(["第" num2str(i) "次適性值: " num2str(fitness(i))])
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
    disp('歷代最佳分數')
    disp(best_score);
    disp('選擇的特徵')
    disp(sort(nov));
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得平均值" + num2str(mean_fitness))
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得變異數" + num2str(var_fitness))
    disp("相同超參數重複建立" + num2str(Repeat_verification)+ "次隨機森林所得標準差" + num2str(std_fitness))
    best_answer=best_params(1:end);  % 由GA找出的最佳的3項RF超參數
    verify=[mean_fitness var_fitness std_fitness];
    best_feature_inx=sort(nov);  % 由GA找出的最佳特徵的編號; 維度=1*指定的特徵數目(numFeats)


    figure(2)
    plot(1:size(design_variables_history,1),design_variables_history(:,end));
    title('適性值疊代過程')
    xlabel('疊代次數')
    ylabel('每次疊代最佳適性值')

end


end