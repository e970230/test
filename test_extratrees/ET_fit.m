function[fitness]=ET_fit(data,label,M,k,nmin,indices)

for sq=1:max(indices)
    % split dataset in two to form the training
    trData.X = data(find(indices~=sq),:);
    trData.Y = label(find(indices~=sq));
    % ... and validation datasets
    valData.X = data(find(indices==sq),:);
    valData.Y = label(find(indices==sq));
    
    
    % Build an ensemble of Extra-Trees and return the predictions on the
    % training dataset.
    [ensemble,trData.YHAT] = buildAnEnsemble(M,k,nmin,[trData.X,trData.Y],0);
    
    % Run the ensemble on a validation dataset
    valData.YHAT = predictWithAnEnsemble(ensemble,[valData.X,valData.Y],0);
    
    fitness(sq) = mean((valData.YHAT - valData.Y).^2);
end
fitness=mean(fitness);
end