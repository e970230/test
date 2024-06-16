clc
clear
close all


load("feature_dataset_heavy.mat")


% n  = 500;
% X = randn(n,3);
% Y = 0.5 * X(:,1) - 0.2 * X(:,2) + 0.1 * X(:,3) + 0.05 * randn(n,1);

X= feature_dataset(:,2:end);
Y= feature_dataset(:,1);

indices = crossvalind('Kfold',Y,5);

% split dataset in two to form the training
trData.X = X(find(indices~=5),:);
trData.Y = Y(find(indices~=5));
% ... and validation datasets
valData.X = X(find(indices==5),:);
valData.Y = Y(find(indices==5));

% Set the parameters for the MATLAB_ExtraTrees aglorithm
M    =  2; % Number of Extra_Trees in the ensemble
k    =  3;  % Number of attributes selected to perform the random splits 
            % 1 <k <= total number of attributes 
nmin =  10;  % Minimum number of points for each leaf

% Build an ensemble of Extra-Trees and return the predictions on the
% training dataset.
[ensemble,trData.YHAT] = buildAnEnsemble(M,k,nmin,[trData.X,trData.Y],0);

% Run the ensemble on a validation dataset
valData.YHAT = predictWithAnEnsemble(ensemble,[valData.X,valData.Y],0);

fitness = mean((valData.YHAT - valData.Y).^2)

% Plot the results
figure;
subplot(211)
plot(trData.Y,'.-r'); hold on; plot(trData.YHAT,'o-b');
xlabel('time'); ylabel('output');
legend('real output','predicted output');
title('Real vs predicted output on the TRAINING dataset');
subplot(212)
plot(valData.Y,'.-r'); hold on; plot(valData.YHAT,'o-b');
xlabel('time'); ylabel('output');
legend('real output','predicted output');
title('Real vs predicted output on the VALIDATION dataset');
