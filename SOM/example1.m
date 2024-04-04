clear
clc
close all

% load training data; 2 fatures, 1000 samples.
data = simplecluster_dataset;

%%
% rng(5); % 設置隨機數種子，以確保結果可重複
% 
% % 生成第一個數據集：x 範圍 -10 到 -5，y 範圍 -10 到 -5
% data1 = [(-5 + (-10 + 5) * rand(1, 100)); (-5 + (-10 + 5) * rand(1, 100))];
% 
% % 生成第二個數據集：x 範圍 -10 到 -5，y 範圍 5 到 10
% data2 = [(-5 + (-10 + 5) * rand(1, 100)); (5 + (10 - 5) * rand(1, 100))];
% 
% % 生成第三個數據集：x 範圍 5 到 10，y 範圍 5 到 10
% data3 = [(5 + (10 - 5) * rand(1, 100)); (5 + (10 - 5) * rand(1, 100))];
% 
% % 生成第四個數據集：x 範圍 5 到 10，y 範圍 -10 到 -5
% data4 = [(5 + (10 - 5) * rand(1, 100)); (-5 + (-10 + 5) * rand(1, 100))];
% % 整合成一個矩陣
% data = [data1 data2 data3 data4];

%%

figure(1) % scatter plot of original dataset
plot(data(1,:),data(2,:),'bo')
xlabel('feature 1')
ylabel('feature 2')
title('input data (before z-score)')


% normalized the training dataset by z-score
dim=2;
%returns the z-score along dimension dim. 
%For example, normalize(A,2) normalizes each row.
x = normalize(data,dim); % dataset after z-score

figure(2) % scatter plot of dataset after z-score
plot(x(1,:),x(2,:),'bo')
xlabel('feature 1')
ylabel('feature 2')
title('input data (after z-score)')


% setting parameters of SOM training
dimensions=[10 10];  %dimensions setting in SOM: 10 by 10 network
numNeuron=dimensions(1)*dimensions(2);   %number of neurons in SOM
coverSteps=100;    %Number of training steps (default=100)
initNeighbor=5;    %Initial neighborhood size, default=3
topologyFcn='hextop';   %Layer topology function: hexagon
distFcn='dist';   %Neuron distance function: Euclidean distance

% train SOM
net=selforgmap(dimensions,coverSteps,initNeighbor,topologyFcn,distFcn);
SOM = train(net,x);   %train SOM by using dataset x




%%

% Get the weight vector of each neuron of trained SOM
wb=getwb(SOM);
weight=[]; %weight vectors
           %(matrix, dimension: number of features by number of neurons)
for k=1:size(x,1)
    weight=[weight ; (wb(((k-1)*numNeuron+1):numNeuron*k))'];
end


% For a specific input data point, find its BMU in SOM
input_data=x(:,200);   % a specific data point from dataset x
isBMU = SOM(input_data); % find the BMU in SOM
BMU_inx=find(isBMU==1);   % isBMU=1: is BMU ; isBMU=0: not BMU
BMU_vector=weight(:,BMU_inx);  % weight vector of BMU


% Calculating the MQE of this input data point
MQE = sum((BMU_vector-input_data).^2).^0.5;  %Euclidean distance

