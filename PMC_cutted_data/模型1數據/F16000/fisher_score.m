function weight = fisher_score(data, target)
    weight = [];
    all_P2 = [];
    target_type = unique(target);
    for i = 1 : size(data, 2) % 尋訪所有特徵i
        X = data(:, i);
        avg_i = mean(X);
        P1 = 0;
        P2 = 0;
        for j = 1 : length(target_type) % 尋訪所有標籤j
            idx = find(target == target_type(j));
            avg_ij = mean(X(idx));
            L = size(idx, 1);
            v = var(X(idx));
            P1 = P1 + (avg_ij - avg_i)^2;
            P2 = P2 + (L * v);
        end
        all_P2 = [all_P2; P2];
        weight = [weight; P1/P2];
    end

    for i = 1:size(data, 2)
        if all_P2(i) < mean(all_P2)*0.01
            weight(i) = 0;
%             warning(['Variance of feature (', num2str(i), ') is too small.'])
        end
    end
end