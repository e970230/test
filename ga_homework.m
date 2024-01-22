clc
clear
close all

%區間
a=0;
b=32;

M=4;        %染色體數
N=10;       %基因數
% for fi=1:100
%%
%隨機產生一個初始族群(編碼
initial_rand_data = randi([0, 1], M, N);

%轉為10進
for i=1:size(initial_rand_data,1)
    for e=1:size(initial_rand_data,2)
        m(i,e)=2.^(10-e)*initial_rand_data(i,e);
    end
end

true_m_data=sum(m,2); %把值加起來

for i=1:size(true_m_data,1)
    true_m_data(i)=a+true_m_data(i)*((b-a)/(2.^size(initial_rand_data,2)-1));   %正規化
end

true_final_data=true_m_data; %先寫一個多餘的讓下面迴圈能跑

%% 迴圈
%參數解碼計算適性函數
for lterate=1:1000   %疊代次數
    true_m_data=true_final_data;
    for s=1:M
        score_m(s)=true_m_data(s).^2;       %適性函數
    end
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
            Ri_round_data(Ri_round_data==0) = NaN;
            [A,B]=min(Ri_round_data);
            Ri_round_data(B)=Ri_round_data(B)-error_Value;
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
    look_max(lterate)=max(true_final_data);
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