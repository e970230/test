function[best_answer,best_score,verify,sample_index] = ga_mix_tree_answer_Fnc(Population_answer,sample_number,Repeat_verification,trainData, trainLabels, validData, validLabels)
Y_sample=sample_number;
if Y_sample==0
    answer=[];
    final_answer=[];
    for pl=1:size(Population_answer,3)  %將最佳解(適性值、最佳的超參數配合)從相對應的疊代裡撈出來
        [answer(pl),answer_index(pl)]=min(Population_answer(:,end,pl));
        final_answer(pl,:)=Population_answer(answer_index(pl),[1:3 end],pl);
    end
    [final_answer_ture,final_answer_index]=min(final_answer(:,4));  %根據適性值把最好的超參數設定連同適性值分數找出來
    final_answer_input=final_answer(final_answer_index,:);          %存成另一個矩陣
    
    ra_test=Repeat_verification; %重複驗證次數
    litera=size(final_answer_input,1);  %需驗證的答案組數
    for li=1:litera
    
        for i=1:ra_test
            input_numTrees = final_answer_input(1);%森林中樹的數量
            input_MaxNumSplits= final_answer_input(2);%每顆樹最大的分割次數
            input_MinLeafSize= final_answer_input(3);%葉節點最小樣本數
            X=[input_numTrees,input_MaxNumSplits,input_MinLeafSize];
            %將完全一樣的條件建立多次的森林得到浮動的適性值
            fitness(li,i) = RandomForestFitnessBasic(X, trainData, trainLabels, validData, validLabels);
            disp("第" + num2str(li) + "組" + "第" + num2str(i) + "次適性值")
            disp(fitness(li,i))
        end
    
    
    end
    %在利用平均數、變異數、標準差等方法來評判此種超參數設定答案是否會浮動過大
    %進而避免導致出現人品爆棚的答案，誤以為是最佳解
    mean_fitness=mean(fitness,2);
    var_fitness=var(fitness,0,2);
    std_fitness=std(fitness,0,2);
    disp('隨機森林裡決策樹的數量')
    disp(input_numTrees);
    disp('每顆樹最大的分割次數')
    disp(input_MaxNumSplits); 
    disp('葉節點最小樣本數')
    disp(input_MinLeafSize);
    disp('歷代最佳分數')
    disp(final_answer_ture);
    disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得平均值" + num2str(mean_fitness))
    disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得變異數" + num2str(var_fitness))
    disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得標準差" + num2str(std_fitness))
    best_answer=final_answer_input(1:end-1);
    best_score=final_answer_ture;
    verify=[mean_fitness var_fitness std_fitness];
    sample_index=[];
    figure(2)
    plot(1:size(final_answer,1),final_answer(:,end));
    title('適性值疊代過程')
    xlabel('疊代次數')
    ylabel('每次疊代最佳適性值')
else
    answer=[];
    final_answer=[];
    for pl=1:size(Population_answer,3)  %將最佳解(適性值、最佳的超參數配合)從相對應的疊代裡撈出來
        [answer(pl),answer_index(pl)]=min(Population_answer(:,end,pl));
        final_answer(pl,:)=Population_answer(answer_index(pl),[1:3 end],pl);
    end
    
    [final_answer_ture,final_answer_index]=min(final_answer(:,4));  %根據適性值把最好的超參數設定連同適性值分數找出來
    final_answer_input=final_answer(final_answer_index,:);          %存成另一個矩陣
    sample=Population_answer(:,:,final_answer_index);               %找出(選取特徵)出現過最佳解的疊代資訊
    [final_sample,final_sample_index]=min(sample(:,end));           %在此次疊代理將最佳選取特徵的資訊找出來
    final_sample_input_answer=sample(final_sample_index,4:end-2);   %存成另一個答案矩陣
    %% 將得到最優解的超參數進行驗證
    
    ra_test=Repeat_verification; %重複驗證次數
    litera=size(final_answer_input,1);  %需驗證的答案組數
    nov=final_sample_input_answer;  %前面多了無關緊要的3個數字，只是因為nov在讀取時是從第四個開始
    for li=1:litera
    
        for i=1:ra_test
            input_numTrees = final_answer_input(1);%森林中樹的數量
            input_MaxNumSplits= final_answer_input(2);%每顆樹最大的分割次數
            input_MinLeafSize= final_answer_input(3);%葉節點最小樣本數
            X=[input_numTrees,input_MaxNumSplits,input_MinLeafSize];
            %將完全一樣的條件建立多次的森林得到浮動的適性值
            fitness(li,i) = RandomForestFitness(X, trainData, trainLabels, validData, validLabels,nov);
            disp("第" + num2str(li) + "組" + "第" + num2str(i) + "次適性值")
            disp(fitness(li,i))
        end
    
    
    end
    %在利用平均數、變異數、標準差等方法來評判此種超參數設定答案是否會浮動過大
    %進而避免導致出現人品爆棚的答案，誤以為是最佳解
    mean_fitness=mean(fitness,2);
    var_fitness=var(fitness,0,2);
    std_fitness=std(fitness,0,2);
    disp('隨機森林裡決策樹的數量')
    disp(input_numTrees);
    disp('每顆樹最大的分割次數')
    disp(input_MaxNumSplits); 
    disp('葉節點最小樣本數')
    disp(input_MinLeafSize);
    disp('歷代最佳分數')
    disp(final_answer_ture);
    disp('選擇的特徵')
    disp(sort(final_sample_input_answer));
    disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得平均值" + num2str(mean_fitness))
    disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得變異數" + num2str(var_fitness))
    disp("相同超參數重複建立" + num2str(ra_test)+ "次隨機森林所得標準差" + num2str(std_fitness))
    best_answer=final_answer_input(1:end-1);
    best_score=final_answer_ture;
    verify=[mean_fitness var_fitness std_fitness];
    sample_index=sort(final_sample_input_answer);
    figure(2)
    plot(1:size(final_answer,1),final_answer(:,end));
    title('適性值疊代過程')
    xlabel('疊代次數')
    ylabel('每次疊代最佳適性值')

end


end