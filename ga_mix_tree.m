clc
clear
close all

%% 初始設立條件

%03/08更改方向
%訓練資料先抓到剩一半
%把基因演算法換成內建的
%時數基因演算法可以看一下
%只取第一個到第十個特徵看是否能趨於一個答案


%------資料集整理
feature_dataset=load("feature_dataset_heavy.mat");
first_data=feature_dataset.feature_dataset;
bit2int = @(bits) sum(bits .* 2.^(length(bits)-1:-1:0));
rpm_120=first_data(find(first_data==120),:);
rpm_680=first_data(find(first_data==680),:);
rpm_890=first_data(find(first_data==890),:);
rpm_1100=first_data(find(first_data==1100),:);
rpm_1400=first_data(find(first_data==1400),:);
rpm_1500=first_data(find(first_data==1500),:);
rpm_2500=first_data(find(first_data==2500),:);

%各轉速選擇時的隨機亂數
%根據各自轉速的樣本數大小進行隨機亂數排列

rpm_120_rand=randperm(size(rpm_120,1));
rpm_680_rand=randperm(size(rpm_680,1));
rpm_890_rand=randperm(size(rpm_890,1));
rpm_1100_rand=randperm(size(rpm_1100,1));
rpm_1400_rand=randperm(size(rpm_1400,1));
rpm_1500_rand=randperm(size(rpm_1500,1));
rpm_2500_rand=randperm(size(rpm_2500,1));

%選擇需要訓練的各資料樣本數
randomly_selected_samples=100;

% %訓練資料集特徵
% train_data=[rpm_120(rpm_120_rand(1:randomly_selected_samples),2:end);rpm_680(rpm_680_rand(1:randomly_selected_samples),2:end);rpm_890(rpm_890_rand(1:randomly_selected_samples),2:end);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),2:end);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),2:end);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),2:end);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),2:end)];
% %訓練資料集特徵對應分類
% train_data_y=[rpm_120(rpm_120_rand(1:randomly_selected_samples),1);rpm_680(rpm_680_rand(1:randomly_selected_samples),1);rpm_890(rpm_890_rand(1:randomly_selected_samples),1);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),1);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),1);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),1);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),1)];
% %測試資料集
% input_data=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),2:end);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),2:end);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),2:end);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),2:end);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),2:end);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),2:end);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),2:end)];
% %測試資料集解答
% input_data_answer=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),1);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),1);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),1);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),1);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),1);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),1);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),1)];

train_data=[rpm_120(rpm_120_rand(1:randomly_selected_samples),2:11);rpm_680(rpm_680_rand(1:randomly_selected_samples),2:11);rpm_890(rpm_890_rand(1:randomly_selected_samples),2:11);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),2:11);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),2:11);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),2:11);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),2:11)];
train_data_y=[rpm_120(rpm_120_rand(1:randomly_selected_samples),1);rpm_680(rpm_680_rand(1:randomly_selected_samples),1);rpm_890(rpm_890_rand(1:randomly_selected_samples),1);rpm_1100(rpm_1100_rand(1:randomly_selected_samples),1);rpm_1400(rpm_1400_rand(1:randomly_selected_samples),1);rpm_1500(rpm_1500_rand(1:randomly_selected_samples),1);rpm_2500(rpm_2500_rand(1:randomly_selected_samples),1)];
input_data=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),2:11);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),2:11);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),2:11);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),2:11);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),2:11);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),2:11);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),2:11)];
input_data_answer=[rpm_120(rpm_120_rand(randomly_selected_samples+1:end),1);rpm_680(rpm_680_rand(randomly_selected_samples+1:end),1);rpm_890(rpm_890_rand(randomly_selected_samples+1:end),1);rpm_1100(rpm_1100_rand(randomly_selected_samples+1:end),1);rpm_1400(rpm_1400_rand(randomly_selected_samples+1:end),1);rpm_1500(rpm_1500_rand(randomly_selected_samples+1:end),1);rpm_2500(rpm_2500_rand(randomly_selected_samples+1:end),1)];



interval_matrix=[1 40;1 20;1 20];%三種未知數的區間矩陣
P = 3;         %未知數
M = 50;         %染色體數
N1= 10;        %單個基因數
N = N1 * P;    %總基因數



% 隨機產生一個初始族群(編碼
initial_rand_data = randi([0, 1], M, N);


% 合併染色體成三個未知數
merged_chromosomes = reshape(initial_rand_data', N1, P, M);
merged_chromosomes = permute(merged_chromosomes, [2, 1, 3]);

% 轉為10進制
true_m_data = zeros(M, P);
for i = 1:M
    for e = 1:P
        true_m_data(i, e) = bit2int(merged_chromosomes(e, :, i));
    end
end

% 正規化之後四捨五入
for nV=1:P  %分別對三個未知數進行各自區間的正規化
    normalization_Value(:,nV) = round(interval_matrix(nV,1) + true_m_data(:,nV) * ...
        ((interval_matrix(nV,2) - interval_matrix(nV,1)) / ((2^(N/P)) - 1)));
end



normalization_Value_end=normalization_Value;
P2_data=initial_rand_data;
lterate=300; %疊代次數

for nol=1:lterate   %疊代次數


    normalization_Value=normalization_Value_end;%10進制數值繼承
    initial_rand_data=P2_data;  %2進制數值繼承



    for tr=1:M
        % 建立一個 TreeBagger 模型
        input_numTrees = normalization_Value(tr,1);%隨機森林裡決策樹的數量
        input_MinLeafSize=normalization_Value(tr,2);%葉節點最小樣本數
        input_MaxNumSplits=normalization_Value(tr,3);%每顆樹最大的分割次數
        treeBaggerModel = TreeBagger(input_numTrees, train_data, train_data_y,'Method','regression', ...
            'MinLeafSize',input_MinLeafSize,'MaxNumSplits',input_MaxNumSplits);
        predictions = predict(treeBaggerModel, input_data);
        mse_answer(tr)=mse(predictions-input_data_answer);
        score(tr)=1./mse_answer(tr);
    
    end
    
    Ri_data = M * (score / sum(score));
    Ri_round_data= round(Ri_data);  %取四捨五入
    
    while sum(Ri_round_data)~=M     %越界條件處理
                error_Value=abs(sum(Ri_round_data)-M);
                if sum(Ri_round_data)<=M
                    [A,B]=max(Ri_round_data);
                    Ri_round_data(B)=Ri_round_data(B)+error_Value;
                    break
                end
                if sum(Ri_round_data) >= M
                    idx_nonzero = find(Ri_round_data ~= 0);  % 找到非零的索引
                    [min_nonzero, min_idx] = min(Ri_round_data(idx_nonzero));  % 找到非零數值中最小值及其索引
                    Ri_round_data(idx_nonzero(min_idx)) = Ri_round_data(idx_nonzero(min_idx)) - error_Value;
                    % 設立條件當今天裡後值為負數則將差值補回，並將差值減到其他更小數裡
                    if min_nonzero - error_Value < 0
                        min_test=0-(min_nonzero - error_Value);
                        Ri_round_data(idx_nonzero(min_idx)) = Ri_round_data(idx_nonzero(min_idx)) + min_test;
                        
                        idx_nonzero = find(Ri_round_data ~= 0);
                        other_indices = idx_nonzero(idx_nonzero ~= idx_nonzero(min_idx));
                        [min_nonzero, min_idx] = min(Ri_round_data(idx_nonzero));
                        Ri_round_data(idx_nonzero(min_idx)) = Ri_round_data(idx_nonzero(min_idx)) - min_test;
                    end
                    break
                end
                
        end
    new_rand_data = repelem(initial_rand_data, Ri_round_data, 1);%將原2進制數據根據輪盤式選擇的答案進行重新排列
    
    %生成一個隨機產生的配對編號
    startValue_pair = 1;
    endValue_pair = M;
    
    % 生成不重複的隨機配對編號
    random_Value_pair = randperm(endValue_pair - startValue_pair + 1) + startValue_pair - 1;
    
    % 設定範圍和要生成的數量
    range_start = 1;
    range_end = N1;
    interval_size = 3;
    intervals = zeros(M, interval_size);
    for i = 1:M
        % 隨機選擇區間的開始位置
        start_point = randi(range_end - interval_size + 1);
        
        % 生成區間
        current_interval = start_point:start_point + interval_size - 1;
        
        % 將區間存入結果矩陣
        intervals(i, :) = current_interval;
    end
    P_data=new_rand_data;
    %交配
    for pd=1:P                                      %從1到P(計算每個未知數
        P_data_test(:,1:N1)=P_data(:,(pd-1)*N1+1:pd*N1);
            for ci=1:M                              %從1到M(計算每個染色體
                colIndex_start = ci;                %主動選擇第ci的染色體
                colIndex_end = random_Value_pair(ci);    %隨機選到的染色體
                if colIndex_start~=colIndex_end     %如果確認選擇的跟隨機的染色體是不同的才執行
                    tempColumn = P_data_test(colIndex_start,intervals(ci,:));
                    P_data_test(colIndex_start,intervals(ci,:)) = P_data_test(colIndex_end,intervals(ci,:));
                    P_data_test(colIndex_end,intervals(ci,:)) = tempColumn;
                end
            end
       P_data(:,(pd-1)*N1+1:pd*N1)=P_data_test(:,1:N1);
    end
    
    %突變
    mutation_rate = 0.2; % 調整突變率
    mutation_Value = round(M * N * mutation_rate);
    mutation_S=zeros(1,2);
    P2_data=P_data;
    for mv=1:mutation_Value
        randomRowIndex = randi(size(P2_data, 1));
        randomColIndex = randi(size(P2_data, 2));
        if mutation_S==[randomColIndex,randomRowIndex]
            randomRowIndex = randi(size(P2_data, 1));
            randomColIndex = randi(size(P2_data, 2));
        end
        randomValue = P2_data(randomRowIndex, randomColIndex);
        if randomValue==1       %是1的話變0
            P2_data(randomRowIndex, randomColIndex)=0;
        else                    %是0的話變1
            P2_data(randomRowIndex, randomColIndex)=1;
        end
        mutation_S=[randomColIndex,randomRowIndex];
    end
    
    merged_chromosomes_end = reshape(P2_data', N1, P, M);
    merged_chromosomes_end = permute(merged_chromosomes_end, [2, 1, 3]);
    
    % 轉為10進制
    normalization_Value_end = zeros(M, P);
    for i = 1:M
        for e = 1:P
            normalization_Value_end(i, e) = bit2int(merged_chromosomes_end(e, :, i));
        end
    end
    
    % 正規化之後四捨五入
    for nV=1:P
        normalization_Value_end(:,nV) = round(interval_matrix(nV,1) + normalization_Value_end(:,nV) * ...
            ((interval_matrix(nV,2) - interval_matrix(nV,1)) / ((2^(N/P)) - 1)));
    end
    
    for tr=1:M
        % 建立一個 TreeBagger 模型
        input_numTrees = normalization_Value_end(tr,1);%隨機森林裡決策樹的數量
        input_MinLeafSize=normalization_Value_end(tr,2);%葉節點最小樣本數
        input_MaxNumSplits=normalization_Value_end(tr,3);%每顆樹最大的分割次數
        treeBaggerModel = TreeBagger(input_numTrees, train_data, train_data_y,'Method','regression', ...
            'MinLeafSize',input_MinLeafSize,'MaxNumSplits',input_MaxNumSplits);
        predictions = predict(treeBaggerModel, input_data);
        mse_answer_end(tr)=mse(predictions-input_data_answer);
        score_end(tr)=1./mse_answer_end(tr);
    
    end
    
    [score_answer,score_answer_index]=max(score_end);
    final_answer=normalization_Value_end(score_answer_index,:);
    
    if nol==1               %第一次疊代將歷代最優解存起來
        best_answer=final_answer;
        best_score=score_answer;
    end
    if score_answer>best_score  %當這次分數比歷代最優解還高時將此次答案儲存下來
        best_answer=final_answer;
        best_score=score_answer;
    end
%---------------------------------驗證區域    

    ture_score=mse(input_data_answer);
    input_numTrees = final_answer(1,1);%森林中樹的數量
    input_MinLeafSize=final_answer(1,2);%葉節點最小樣本數
    input_MaxNumSplits=final_answer(1,3);%每顆樹最大的分割次數
    treeBaggerModel_verifications_mode = TreeBagger(input_numTrees, train_data, train_data_y,'Method','regression', ...
        'MinLeafSize',input_MinLeafSize,'MaxNumSplits',input_MaxNumSplits);
    predictions = predict(treeBaggerModel, input_data);
    mse_answer(nol)=mse(predictions-input_data_answer);
    ture(nol)=mse_answer(nol)./ture_score;
        
        

    if min(mse_answer)<=200
        break
    end


end

%%


plot(1:nol,mse_answer)

disp('隨機森林裡決策樹的數量')
disp(final_answer(1,1));
disp('葉節點最小樣本數')
disp(final_answer(1,2));
disp('每顆樹最大的分割次數')
disp(final_answer(1,3));
disp('疊代次數')
disp(nol)

