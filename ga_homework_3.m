clc
clear
close all
load No2_curve_data.mat
% objectiveFunction = @(x,t) x(1).*exp(x(2).*t)+x(3);
% 定義bit2int函数
bit2int = @(bits) sum(bits .* 2.^(length(bits)-1:-1:0));

% 區間
a = -5;
b = 5;
P = 3;         % 未知數
M = 8;         % 染色體數
N1= 15;         %單個基因數
N = N1 * P;     % 總基因數

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

% 正規化
true_m_data = a + true_m_data * ((b - a) / ((2^(N/P)) - 1));

% 
true_m_data_end=true_m_data; %先寫一個多餘的讓下面迴圈能跑
P2_data=initial_rand_data;
for lterate=1:1000   %疊代次數
    
    true_m_data=true_m_data_end;    %10進制數值繼承
    initial_rand_data=P2_data;      %2進制數值繼承
    true_m_data = reshape(true_m_data', [], 1);
    score_m = 1./(sum((y1test(true_m_data(1:P:end), true_m_data(2:P:end), t, true_m_data(3:P:end)) - y).^2, 2));

    %%
    %輪盤式選擇(複製
    Ri_data = M * (score_m / sum(score_m));
    Ri_round_data= round(Ri_data);  %取四捨五入
    while sum(Ri_round_data)~=M     %越界條件處理
            error_Value=abs(sum(Ri_round_data)-M);
            if sum(Ri_round_data)<=M
                [A,B]=max(Ri_round_data);
                Ri_round_data(B)=Ri_round_data(B)+error_Value;
                break
            end
            if sum(Ri_round_data) >= M
                idx_nonzero = find(Ri_round_data ~= 0);  % 找到非零元素的索引
                [min_nonzero, min_idx] = min(Ri_round_data(idx_nonzero));  % 找到非零元素中最小值及其索引
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
    % 使用 repelem 生成新的交配池
    new_rand_data = repelem(initial_rand_data, Ri_round_data, 1);
    
    %生成一個隨機產生的配對編號
    startValue = 1;
    endValue = M;
    
    % 生成不重复的隨機配對編號
    random_Value = randperm(endValue - startValue + 1) + startValue - 1;
    
    
    % 設定範圍和要生成的數量
    range_start = 1;
    range_end = N1;
    % interval_size = 3;
    if lterate<=1000
        interval_size = 3;
    end
    if lterate>1000 && lterate<=10000
        interval_size = 2;
    end
    if lterate>10000
        interval_size=1;
    end
    num_intervals = M;
    % 生成區間
    intervals = zeros(num_intervals, interval_size);
    for i = 1:num_intervals
        % 隨機選擇區間的開始位置
        start_point = randi(range_end - interval_size + 1);
        
        % 生成區間
        current_interval = start_point:start_point + interval_size - 1;
        
        % 將區間存入結果矩陣
        intervals(i, :) = current_interval;
    end
    
    %%
    P_data=new_rand_data;
    %交配
    for pd=1:P
        P_data_test(:,1:N1)=P_data(:,(pd-1)*N1+1:pd*N1);
            for ci=1:M
                colIndex_start = ci;                %主動選擇第ci的染色體
                colIndex_end = random_Value(ci);    %隨機選到的染色體
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
    
    %%
    %轉為10進
    merged_chromosomes_end = reshape(P2_data', N1, P, M);
    merged_chromosomes_end = permute(merged_chromosomes_end, [2, 1, 3]);
    
    % 轉為10進制
    true_m_data_end = zeros(M, P);
    for i = 1:M
        for e = 1:P
            true_m_data_end(i, e) = bit2int(merged_chromosomes_end(e, :, i));
        end
    end
    % 正規化
    true_m_data_end = a + true_m_data_end * ((b - a) / ((2^(N/P)) - 1));
    
    for yt=1:M
        y_test(yt,:)=y1test(true_m_data_end(yt,1),true_m_data_end(yt,2),t,true_m_data_end(yt,3));
    end
    y_test_final=sum((y_test-y).^2,2);
    [y_min,y_index]=min(y_test_final);
    look_ymin(lterate)=y_min;
    true_min(lterate,:)=true_m_data_end(y_index,:);
    if y_min<=40        %當最小值達成條件則跳出矩陣
        break
    end
end

%%
[look_min_Vaule,look_min_index]=min(look_ymin);
data_min=true_min(look_min_index,:);
y_answer=y1test(data_min(1,1),data_min(1,2),t,data_min(1,3));
disp(look_min_Vaule)
disp(data_min)

figure(1)
plot(t,y_answer,'LineWidth',0.5,'Color','R','LineStyle','none','Marker','*')
figure(2)
plot(t,y,'LineWidth',0.5,'Color','G','LineStyle','none','Marker','*')
figure(3)
plot(1:lterate,look_ymin,'LineWidth',0.5,'Color','G','LineStyle','none','Marker','*')