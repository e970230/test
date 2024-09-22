%_________________________________________________________________________%
%  測試

% You can simply define your cost in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of generations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single number numbers

% To run WOA: [Best_score,Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%__________________________________________


clc
clear 
close all

load ("PSO_exercise_dataset.mat")
SearchAgents_no=100; % Number of search agents


Max_iteration=500; % Maximum numbef of iterations



fobj =@(x) CoFnc(y3test(x(1),x(2),t,x(3),x(4)),y);  %costfunction


lb = [0 0 0 0];     %costfunction所有解答的下限
ub = [5 10 20 5];   %costfunction所有解答的上限
dim = 4;            %costfunction需要尋找的解答數量



[Best_score,Best_pos,WOA_cg_curve]=WOA_self_ver_01(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);




%%

disp("最佳輸入參數")
disp("a1:"+ num2str(Best_pos(1)) + " a2:"+ num2str(Best_pos(2)) + " a3:"+ num2str(Best_pos(3)) + " a4:"+ num2str(Best_pos(4)))
figure(1)
%plot(BestCost,'LineWidth',2);
semilogy(WOA_cg_curve,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;
figure(2)
hold on
plot(t,y3test(Best_pos(1),Best_pos(2),Best_pos(3),Best_pos(4),t),'LineWidth',0.5,'Color','R','LineStyle','-')
plot(t,y,'LineWidth',0.5,'Color','G','LineStyle','--')
hold off
xlabel("time")
ylabel("y")
legend('預測解','原始最佳解')



