clc
clear
close all

%% 初始設立條件
frist_data=load("PCA_test_data.txt","-ascii");
bit2int = @(bits) sum(bits .* 2.^(length(bits)-1:-1:0));

% 區間
interval_matrix=[10 40;3 8;1 5];%三種未知數的區間矩陣
P = 3;         %未知數
M = 8;         %染色體數
N1= 15;        %單個基因數
N = N1 * P;    %總基因數

startValue = 1;                     %採樣區間隨機起始數值
endValue = 50;                      %採樣區間隨機結束數值
random_Value = randperm(endValue - startValue + 1) + startValue - 1;        %隨機選擇
train_test_Value=50*0.8;                 %訓練數的樣本數，其中0.8代表這次選擇百分之80的數據進行訓練
input_test_Value=50;                     %採樣完後剩餘的數值作為測試數據，代表剩餘百分之20的數據做為測試資料
Sample_type=3;                      %此鳶尾花種類為三種
train_data=zeros(train_test_Value.*Sample_type,size(frist_data,2));

% for id=1:Sample_type
%     train_data_Site_storage(:,:,id)=[frist_data(random_Value(1:train_test_Value).*id,1:4)];
%     train_data_Site_storage_answer(:,:,id)=[frist_data(random_Value(1:train_test_Value).*id,5)];
%     input_data_Site_storage(:,:,id)=[frist_data(random_Value(train_test_Value+1:input_test_Value).*id,1:4)];
%     input_data_Site_storage_answer(:,:,id)=[frist_data(random_Value(train_test_Value+1:input_test_Value).*id,5)];
%     train_data(train_test_Value.*(id-1)+1:train_test_Value.*id,1:4)=train_data_Site_storage(:,:,id);
%     train_data(train_test_Value.*(id-1)+1:train_test_Value.*id,5)=train_data_Site_storage_answer(:,:,id);
% end
train_data=[frist_data((1:40)+50*0,1:4);frist_data((1:40)+50*1,1:4);frist_data((1:40)+50*2,1:4)]; 
train_data_y=[frist_data((1:40)+50*0,5);frist_data((1:40)+50*1,5);frist_data((1:40)+50*2,5)];
input_data=[frist_data((41:50)+50*0,1:4);frist_data((41:50)+50*1,1:4);frist_data((41:50)+50*2,1:4)];
input_data_answer=[frist_data((41:50)+50*0,5);frist_data((41:50)+50*1,5);frist_data((41:50)+50*2,5)];

% input_data=[frist_data(random_Value(train_test_Value+1:input_test_Value),1:4)];         %以目前的條件第121~150項的隨機數
% input_data_answer=[frist_data(random_Value(train_test_Value+1:input_test_Value),5)];
% train_data=[frist_data(random_Value(1:train_test_Value),1:4)];                          %以目前的條件第1~120項的隨機數
% train_data_y=[frist_data(random_Value(1:train_test_Value),5)];


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


insurance_value=0.00001; %以防再答案完全吻合時分母為0的保險
normalization_Value_end=normalization_Value;
P2_data=initial_rand_data;


%%


for lterate=1:100   %疊代次數


    normalization_Value=normalization_Value_end;%10進制數值繼承
    initial_rand_data=P2_data;  %2進制數值繼承

    for tr=1:M
        % 建立一個 TreeBagger 模型
        input_numTrees = normalization_Value(tr,1);%決策數的數量
        input_MinLeafSize=normalization_Value(tr,2);%葉節點最小樣本數
        input_MaxNumSplits=normalization_Value(tr,3);%每顆樹最大的分割次數
        treeBaggerModel = TreeBagger(input_numTrees, train_data, train_data_y,'Method','classification', ...
            'MinLeafSize',input_MinLeafSize,'MaxNumSplits',input_MaxNumSplits);
        predictions = predict(treeBaggerModel, input_data);
        answer_prediceions=str2double(predictions);%由於輸出的答案不是數值是字串，所以將字串轉數值
        score(tr,1)=1/((sum(abs(answer_prediceions-input_data_answer)))+insurance_value);
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
    mutation_rate = 0.3; % 調整突變率
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



    for tre=1:M
        % 建立一個 TreeBagger 模型
        input_numTrees = normalization_Value_end(tre,1);%決策數的數量
        input_MinLeafSize=normalization_Value_end(tre,2);%葉節點最小樣本數
        input_MaxNumSplits=normalization_Value_end(tre,3);%每顆樹最大的分割次數
        treeBaggerModel = TreeBagger(input_numTrees, train_data, train_data_y,'Method','classification', ...
            'MinLeafSize',input_MinLeafSize,'MaxNumSplits',input_MaxNumSplits);
        predictions = predict(treeBaggerModel, input_data);
        answer_prediceions=str2double(predictions);
        score_end(tre,1)=1/((sum(abs(answer_prediceions-input_data_answer)))+insurance_value);
    end
    [score_answer,score_answer_index]=max(score_end);
    final_answer=normalization_Value_end(score_answer_index,:);
    if lterate==1               %第一次疊代將歷代最優解存起來
        best_answer=final_answer;
        best_score=score_answer;
    end
    if score_answer>best_score  %當這次分數比歷代最優解還高時將此次答案儲存下來
        best_answer=final_answer;
        best_score=score_answer;
    end

    if score_answer==1/insurance_value %分數達到理論最大值時跳出
        break
    end

end


%%
disp('最佳解')
disp(final_answer);
disp('最佳分數')
disp(best_score)
% view(treeBaggerModel.Trees{1},Mode="graph")