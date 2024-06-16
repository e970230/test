function[fitness]=ET_fit(data,label,M,k,nmin,indices)

% split dataset in two to form the training
trData.X = data(find(indices~=5),:);
trData.Y = label(find(indices~=5));
% ... and validation datasets
valData.X = data(find(indices==5),:);
valData.Y = label(find(indices==5));


% Build an ensemble of Extra-Trees and return the predictions on the
% training dataset.
[ensemble,trData.YHAT] = buildAnEnsemble(M,k,nmin,[trData.X,trData.Y],0);

% Run the ensemble on a validation dataset
valData.YHAT = predictWithAnEnsemble(ensemble,[valData.X,valData.Y],0);

fitness = mean((valData.YHAT - valData.Y).^2);

end