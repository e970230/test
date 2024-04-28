function [state, options, optchanged] = gaoutputfunction(options, state, flag)
    % 定義history用來存儲每次跌代的答案和分數
    
    persistent history
    Generation=state.Generation;  %從state裡獲取現在疊代次數的資訊
    if Generation==0
        history=[];     %第一次疊代時將history重製
    end
    % 在每次跌代(iteration)時，保存當前的Generation、Best和BestScore
    if strcmpi(flag, 'iter')
        % 將當前的Generation、Best和BestScore添加到history的最後一個元素中
        history(:,1:size(state.Population,2),Generation) = state.Population;                    %每個染色體各自的超參數答案(包含選擇的特徵)
        history(:,size(state.Population,2)+1,Generation) = state.Best(state.Generation);        %最佳分數
        history(:,size(state.Population,2)+2,Generation) = state.Score;                         %每個染色體各自的分數
        
    end
    
    % 設置optchanged為false，表示未更改選項
    optchanged = false;
    
    if state.Generation>=1
    state.Population_answer(:,:,Generation) = history(:,:,Generation);
    assignin('base', 'Population_answer', state.Population_answer);   %將副程式算出的值顯示在主程式的workspace上
    end


end
