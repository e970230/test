clc
clear
close all

tic
load PSO_exercise_dataset.mat


%%


% 設定基因演算法的初始參數`
PopulationSize=100;     %族群大小意旨染色體數目
% FitnessLimit=5;      %目標函數跳出門檻(小於此值則演算法結束
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', PopulationSize, ...
    'Generations',1000,'OutputFcn',@gaoutputfunction,'CrossoverFraction',0.7,'MaxStallGenerations',inf);
%'iter'顯示出每次疊代的詳細資訊
%'PlotFcn'畫圖指令
% gaplotbestf紀錄每次跌代中最佳答案的分數(fitness)
% gaplotbestindiv最佳解答(x)
% gaplotexpectation每次跌代的期望值
% Generations疊代次數
% OutputFun輸出資訊客製化
% CrossoverFraction更改交配率

% 定義要優化的參數範圍
numVariables = 4; 
lb = [0 0 0 0];  % 下限
ub = [5 10 20 5]; % 上限
Iteration=10;
Population_it_answer={};
for it=1:Iteration
    % 定義目標函數
    CostFunction= @(x) CoFnc(y3test(x(1),x(2),x(3),x(4),t),y);        % Cost Function
    % intcon=1:4;
    
    
    % 使用基因演算法搜索最佳參數
    [x,fval,exitflag,output,population,scores] = ga(CostFunction, numVariables, [], [], [], [], lb, ub, [],options);
    
    %[x]解答
    %[fval]分數
    %[exitflag]結束運算的原因
    %[output]輸出運算法的結構
    %[population]最後一次跌代所有染色體的解答
    %[scores]最後一次跌代所有染色體的分數
    %Func-count調動目標函數的次數
    %stall generations疊代沒有優化的次數
    
    % delete(gcp);
    
    %%
    answer=[];
    final_answer=[];
    
    for pl=1:size(Population_answer,3)
        [answer(pl),answer_index(pl)]=min(Population_answer(:,6,pl));
        final_answer(pl,:)=Population_answer(answer_index(pl),[1:4 6],pl);
    end
    
    [final_answer_ture,final_answer_index]=min(final_answer(:,5));
    final_answer_input(it,:)=final_answer(final_answer_index,:);
    Population_it_answer{it}=Population_answer;

end
%%
mean_answer=mean(final_answer_input(:,5))
var_answer=var(final_answer_input(:,5))
mse_answer=mse(final_answer_input(:,5))


toc
%%
disp('歷代最佳適性值')
disp(final_answer_ture);
figure(1)
plot(1:size(final_answer,1),final_answer(:,5));
xlabel('疊代次數')
ylabel('每次疊代最佳MSE')
figure(2)
hold on
plot(t,y3test(final_answer_input(1),final_answer_input(2),final_answer_input(3),final_answer_input(4),t),'LineWidth',0.5,'Color','R','LineStyle','-')
plot(t,y,'LineWidth',0.5,'Color','G','LineStyle','--')
hold off
xlabel("time")
ylabel("y")
legend('預測解','原始最佳解')
figure(3)
bar(1:length(final_answer_input),final_answer_input(:,5))
title('相同初始條件多次執行測試')
xlabel('執行次數')
ylabel('適性值分數')
disp(['平均值' +num2str(mean(final_answer_input(:,5)))])
disp(['變異數' +num2str(var(final_answer_input(:,5)))])

toc
