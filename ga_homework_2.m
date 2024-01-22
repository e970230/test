clc
clear
close all
load No2_curve_data.mat
% objectiveFunction = @(x,t) x(1).*exp(x(2).*t)+x(3);


% 區間
a = -5;
b = 5;
P = 3;         % 未知數
M = 4;         % 染色體數
N = 5 * P;     % 基因數

% 隨機產生一個初始族群(編碼
initial_rand_data = randi([0, 1], M, N);

% 合併染色體成三個未知數
merged_chromosomes = reshape(initial_rand_data', 5, P, M);
merged_chromosomes = permute(merged_chromosomes, [2, 1, 3]);

% 轉為10進制
true_m_data = zeros(M, P);
for i = 1:M
    for e = 1:P
        true_m_data(i, e) = bi2de(merged_chromosomes(e, :, i)) / (2^(N/P) - 1);
    end
end

% 正規化
true_m_data = a + true_m_data * ((b - a) / ((2^(N/P)) - 1));


true_final_data=true_m_data; %先寫一個多餘的讓下面迴圈能跑

%% 迴圈
%參數解碼計算適性函數
for lterate=1:100   %疊代次數
    true_m_data=true_final_data;
    for s=1:M
        score_m(s)=sum(true_m_data(1,s).*exp(true_m_data(2,s).*t)+true_m_data(3,s));       %適性函數
    end
    %%
    %輪盤式選擇(複製
    for ri=1:M
        Ri_data(ri)=M.*(score_m(ri)/sum(score_m));  %計算Ri值
    end
    
    Ri_round_data= round(Ri_data);  %取四捨五入
    while sum(Ri_round_data)~=M     %越界條件處理
        error_Value=abs(sum(Ri_round_data)-M);
        if sum(Ri_round_data)<=M
            [A,B]=max(Ri_round_data);
            Ri_round_data(B)=Ri_round_data(B)+error_Value;
            break
        end
        if sum(Ri_round_data)>=M
            [A,B]=max(Ri_round_data);
            Ri_round_error_data=sort(Ri_round_data,'descend');
            Ri_round_error_data(2)=Ri_round_data(2)-error_Value;
            Ri_round_data=Ri_round_error_data;
            break
        end
        
    end
    k=1;
    %創建一個交配池
    new_rand_data=[];
    for rn=1:size(Ri_round_data,2)
        for rj=1:Ri_round_data(rn)
            new_rand_data(k,:)=initial_rand_data(rn,:); %把已經知道複製個數的染色體放入新的矩陣裡
            k=k+1;
        end
        if size(new_rand_data,1)==4
            break
        end
    end
 
    %生成一個隨機產生的配對編號
    startValue = 1;
    endValue = M;
    
    % 生成不重复的随机排列
    random_Value = randperm(endValue - startValue + 1) + startValue - 1;
    
    %生成一個隨機產生交配點
    start_copulation = 1;
    end_copulation = N;
    rand_copulation = randi([start_copulation, end_copulation], startValue, endValue);
    
    P_data=new_rand_data;
    
    %交配
    for ci=1:M
        colIndex_start = ci;                %主動選擇第ci的染色體
        colIndex_end = random_Value(ci);    %隨機選到的染色體
        if colIndex_start~=colIndex_end     %如果確認選擇的跟隨機的染色體是不同的才執行
            tempColumn = P_data(colIndex_start,rand_copulation(ci));
            P_data(colIndex_start,rand_copulation(ci)) = P_data(colIndex_end,rand_copulation(ci));
            P_data(colIndex_end,rand_copulation(ci)) = tempColumn;
        end
    end
    
    
    
    %突變
    mutation_Value=M*N/10;      %突變數(在所有基因因子裡隨機挑選數個進行突變
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
        mutation_S(mv,:)=[randomColIndex,randomRowIndex];
    end
    %%
    %轉為10進
    %最後計算得出來的因子轉出來的數值
    for i=1:M
        for e=1:N
            m_end(i,e)=2.^(10-e).*P2_data(i,e);
        end
    end
    
    true_final_data=sum(m_end,2);
    
    for i=1:size(true_final_data,1)
        true_final_data(i)=a+true_final_data(i)*((b-a)/(2.^size(P2_data,2)-1));
    end
    if max(true_final_data)>=31.5       %如果大於31.5的話則直接中止迴圈
        break
    end
end

answer=max(true_final_data);


%     if answer>=31.5
%         break
%     end
% end

disp('最終解答')
disp(answer)