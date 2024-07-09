%% 測試diff評判差值過大情況

clc
clear
close all
%--------------------------------------------------------------------------
%創建模擬訊號
x=1:1000;
y=0.005*x+0.005;

%添加測試訊號
y_test=7*ones(1,100);
y(201:300)=y_test;
y(401:500)=y_test;
y(701:800)=y_test;

[output]=diff_signal(y,30,0.1);

figure(1)
plot(x,y)
xlabel('sample')
ylabel('訊號')
