function [state, options, optchanged] = gaoutputfunction(options, state, flag)
% ga.m的'OutputFcn'設定了自訂的gaoutputfunction.m, ga將於每次疊代時呼叫並執行此function
% 實際上, 在本應用中, GA的運算結果並不是由ga.m的輸出x與fval得到, 而是由ga中的'OutputFcn'所設定
% 的自訂gaoutputfunction.m, 由ga的每次疊代過程的資訊中所獲得

% last modification: 2024/05/07


    
persistent history   %以history儲存GA於各次疊代時, 各染色體的設計變數值(包含超參數與GA挑選的特徵的編號),
                     %與fitness(MSE值)

Generation=state.Generation;  %從state裡獲取當前的疊代次數 (當前是第幾次疊代)

if Generation==0
    history=[];     %第一次疊代時將history重置
end

    
% 在每次疊代時，保存當前的Generation、Best和BestScore
if strcmpi(flag, 'iter')  %假如當前ga還處於疊代過程中
    % 將當前的Generation、Best和BestScore添加到history的最後一個元素中
    history(:,1:size(state.Population,2),Generation) = state.Population;  %每個染色體的設計變數值(包含超參數與GA挑選的特徵的編號); 維度=族群大小*GA的設計變數的數量
    history(:,size(state.Population,2)+1,Generation) = state.Score;        %每個染色體的fitness(MSE值); 維度=族群大小*1
        
end
    
% 設置optchanged為false，表示未更改選項
optchanged = false;
    
if state.Generation>=1
    state.Population_answer(:,:,Generation) = history(:,:,Generation);

    % Population_answer: 儲存了GA於各次疊代過程中, 各染色體的設計變數值(包含超參數與GA挑選的特徵的編號), 
    %                    與fitness(MSE值);
    %                    維度 = 族群大小*GA的設計變數的數量(=3項超參數+numFeats個特徵)*總疊代次數

    assignin('base', 'Population_answer', state.Population_answer);   %將副程式算出的值顯示在主程式的workspace上
end


end
