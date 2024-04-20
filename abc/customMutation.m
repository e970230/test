function mutatedChildren = customMutation(parents, options, nvars, FitnessFcn, state, thisScore, thisPopulation)
%     mutationRate = 30;  % 自定義的突變率，這裡設為 30%
    
    % 對每個染色體進行突變
    for i = 1:length(parents)
        child = thisPopulation(parents(i), :);
        % 隨機判斷每個變量是否進行突變
        for j = 1:nvars
            child(j) = unifrnd(options.PopInitRange(1,j),options.PopInitRange(2,j),1);
        end
        mutatedChildren(i, :) = child;
    end
end