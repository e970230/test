function [state, options, optchanged] = gaoutputfunction(options, state, flag)
    % 定義history用來存儲每次跌代的答案和分數
    persistent history
        
    % 在每次跌代(iteration)時，保存當前的Generation、Best和BestScore
    if strcmpi(flag, 'iter')
        % 將當前的Generation、Best和BestScore添加到history的最後一個元素中
        history(end+1).Generation = state.Generation;   %跌代次數
        history(end).Population = state.Population;     %每個染色體各自的答案
        history(end).Best = state.Best;                 %最佳分數
        history(end).Score = state.Score;               %每個染色體各自的分數
    end
    
    % 設置optchanged為false，表示未更改選項
    optchanged = false;
    
    % 根據不同的flag顯示相應的信息
    switch flag
        case 'init'
            disp('Starting the algorithm');
        case {'interrupt'}
            disp('Subproblem for nonlinear constraints');
        case 'done'
            disp('Performing final task');
    end
    
    % 將history存儲在state的UserData字段中
    state.UserData = history;
    
    % 將Generation、Best和Score轉換為一個大矩陣並儲存
    numIterations = length(history);
    Population_answer = [];
    for i = 1:numIterations
        Population_answer(:, 1:3,i) = history(i).Population;
        Population_answer(:, 4,i) = history(i).Best(i);
        Population_answer(:, 5,i) = history(i).Score;
    end

    state.Population_answer = Population_answer;
    assignin('base', 'Population_answer', Population_answer);   %將副程式算出的值顯示在主程式的workspace上
%     whos Population_answer                                      %顯示Population_answer的信息
end
