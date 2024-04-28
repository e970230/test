function[] = ga_mix_tree_Fnc(ga_input,sample_number,lb_input,ub_input ,trainData, trainLabels, validData, validLabels)

% 設定基因演算法的初始參數
PopulationSize=ga_input(1);     %族群大小意旨染色體數目
Generations=ga_input(2);
CrossoverFraction=ga_input(3);
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'Generations',Generations,'CrossoverFraction', CrossoverFraction,'OutputFcn',@gaoutputfunction);

Y_sample=sample_number;    %選取要得樣本數量
% 定義要優化的參數範圍
Y_answer=3;     %要決定的超參數數量
numVariables = Y_answer+Y_sample; % 三個參數(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數)加上選取的樣本數量
intcon=1:numVariables;  %整數變數的數量


lb_sample=ones(1,Y_sample); %每個選取樣本的下限
ub_sample=size(trainData,2)*ones(1,Y_sample); %每個選取樣本的上限
lb = [lb_input lb_sample];  % 下限
ub = [ub_input ub_sample]; % 上限




%%
if Y_sample==0
    fitnessFunction = @(x) RandomForestFitnessBasic(x, trainData, trainLabels, validData, validLabels);
else
    % 定義目標函數，x裡包含所有的猜值(森林裡決策樹的數量、每顆樹最大的分割次數、葉節點最小樣本數、選取的樣本
    fitnessFunction = @(x) RandomForestFitness(x, trainData, trainLabels, validData, validLabels,x(Y_answer+1:end));
end
% 使用基因演算法搜索最佳參數
[x,fval] = ga(fitnessFunction, numVariables, [], [], [], [], lb, ub, [],intcon,options);

%[x]解答
%[fval]分數
%[exitflag]結束運算的原因
%[output]輸出運算法的結構
%[population]最後一次跌代所有染色體的解答
%[scores]最後一次跌代所有染色體的分數
%Func-count調動目標函數的次數
%stall generations疊代沒有優化的次數



end