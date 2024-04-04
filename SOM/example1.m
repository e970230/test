clear
clc
close all

% load training data; 2 fatures, 1000 samples.
% data = simplecluster_dataset;

load("PCA_test_data.txt");
data=PCA_test_data(:,1:4);
data=data';



% figure(1) % scatter plot of original dataset
% plot(data(1,:),data(2,:),'bo')
% xlabel('feature 1')
% ylabel('feature 2')
% title('input data (before z-score)')


% normalized the training dataset by z-score
dim=2;
%returns the z-score along dimension dim. 
%For example, normalize(A,2) normalizes each row.
x = normalize(data,dim); % dataset after z-score

% figure(2) % scatter plot of dataset after z-score
% plot(x(1,:),x(2,:),'bo')
% xlabel('feature 1')
% ylabel('feature 2')
% title('input data (after z-score)')


% setting parameters of SOM training
dimensions=[8 8];  %dimensions setting in SOM: 10 by 10 network
numNeuron=dimensions(1)*dimensions(2);   %number of neurons in SOM
coverSteps=100;    %Number of training steps (default=100)
initNeighbor=1;    %Initial neighborhood size, default=3
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

%%
% For a specific input data point, find its BMU in SOM
input_data=x(:,4);   % a specific data point from dataset x
isBMU = SOM(input_data); % find the BMU in SOM
BMU_inx=find(isBMU==1);   % isBMU=1: is BMU ; isBMU=0: not BMU
BMU_vector=weight(:,BMU_inx);  % weight vector of BMU


% Calculating the MQE of this input data point
MQE = sum((BMU_vector-input_data).^2).^0.5;  %Euclidean distance

