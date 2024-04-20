clc;
clear;
close all;

%% Problem Definition
load PSO_exercise_dataset.mat
tic

CostFunction= @(x) CoFnc(y3test(x(1),x(2),x(3),x(4),t),y);        % Cost Function

nVar=4;             % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size

VarMin=[0 0 0 0];         % Decision Variables Lower Bound
VarMax=[5 10 20 5];         % Decision Variables Upper Bound

%% ABC Settings

MaxIt=1000;              % Maximum Number of Iterations

nPop=100;               % Population Size (Colony Size)

nOnlooker=nPop*0.5;         % Number of Onlooker Bees

L=round(0.5*nVar*nPop); % Abandonment Limit Parameter (Trial Limit)

a=1;                    % Acceleration Coefficient Upper Bound

for IT=1:10
    %% Initialization
    % Empty Bee Structure
    empty_bee.Position=[];
    empty_bee.Cost=[];
    
    % Initialize Population Array
    pop=repmat(empty_bee,nPop,1);
    
    % Initialize Best Solution Ever Found
    BestSol.Cost=inf;
    
    % Create Initial Population
    for i=1:nPop
        for nv=1:nVar
            pop(i).Position(nv)=unifrnd(VarMin(nv),VarMax(nv),1);   %針對每一個參數的上下限進行隨機取值
        end
        pop(i).Cost=CostFunction(pop(i).Position);
        if pop(i).Cost<=BestSol.Cost
            BestSol=pop(i);
        end
    end
    
    % Abandonment Counter
    C=zeros(nPop,1);
    % Array to Hold Best Cost Values
    BestCost=zeros(MaxIt,1);



%% ABC Main Loop



    for it=1:MaxIt
        
        % Recruited Bees
        for i=1:nPop
            
            % Choose k randomly, not equal to i
            K=[1:i-1 i+1:nPop];
            k=K(randi([1 numel(K)]));
            
            % Define Acceleration Coeff.
            phi=a*unifrnd(-1,+1,VarSize);
            
            % New Bee Position
            newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
            
            % Evaluation
            newbee.Cost=CostFunction(newbee.Position);
            
            % Comparision
            if newbee.Cost<=pop(i).Cost
                pop(i)=newbee;
            else
                C(i)=C(i)+1;
            end
            
        end
        
        % Calculate Fitness Values and Selection Probabilities
        F=zeros(nPop,1);
        Sum_Cost = sum([pop.Cost]);
        for i=1:nPop
            
            F(i) = inv(pop(i).Cost/Sum_Cost); % Convert Cost to Fitness
        end
        P=F/sum(F);
        
        % Onlooker Bees
        for m=1:nOnlooker
            
            % Select Source Site
            i=RouletteWheelSelection(P);
            
            % Choose k randomly, not equal to i
            K=[1:i-1 i+1:nPop];
            k=K(randi([1 numel(K)]));
            
            % Define Acceleration Coeff.
            phi=a*unifrnd(-1,+1,VarSize);
            
            % New Bee Position
            newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
            
            % Evaluation
            newbee.Cost=CostFunction(newbee.Position);
            
            % Comparision
            if newbee.Cost<=pop(i).Cost
                pop(i)=newbee;
            else
                C(i)=C(i)+1;
            end
            
        end
        
        % Scout Bees
        for i=1:nPop
            if C(i)>=L
                for nv=1:nVar
                    pop(i).Position(nv)=unifrnd(VarMin(nv),VarMax(nv),1);
                end
                pop(i).Cost=CostFunction(pop(i).Position);
                C(i)=0;
            end
            %有可能發生懲罰值沒到但已經有參數先超出設定上下限
            %所以多一個偵查的條件將超出範圍的答案重新代替
            for nv=1:nVar
                if pop(i).Position(nv)<VarMin(nv) || pop(i).Position(nv)>VarMax(nv)
                    pop(i).Position(nv)=unifrnd(VarMin(nv),VarMax(nv),1);
                    C(i)=0;
                end
            end
        end
        
        % Update Best Solution Ever Found
        for i=1:nPop
            if pop(i).Cost<=BestSol.Cost
                BestSol=pop(i);
            end
        end
        
        % Store Best Cost Ever Found
        BestCost(it)=BestSol.Cost;
        
        % Display Iteration Information
        disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
        
    end
    answer_Position(IT,:)=BestSol.Position;
    answer_Cost(IT)=BestSol.Cost;
end
%% Results

disp("最佳輸入參數")
disp("a1:"+ num2str(BestSol.Position(1)) + " a2:"+ num2str(BestSol.Position(2)) + " a3:"+ num2str(BestSol.Position(3)) + " a4:"+ num2str(BestSol.Position(4)))
figure(1)
%plot(BestCost,'LineWidth',2);
semilogy(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;
figure(2)
hold on
plot(t,y3test(BestSol.Position(1),BestSol.Position(2),BestSol.Position(3),BestSol.Position(4),t),'LineWidth',0.5,'Color','R','LineStyle','-')
plot(t,y,'LineWidth',0.5,'Color','G','LineStyle','--')
hold off
xlabel("time")
ylabel("y")
legend('預測解','原始最佳解')
figure(3)
bar(1:length(answer_Cost),answer_Cost(1,:))
title('相同初始條件多次執行測試')
xlabel('執行次數')
ylabel('適性值分數')
disp(['平均值' +num2str(mean(answer_Cost))])
disp(['變異數' +num2str(var(answer_Cost))])
toc