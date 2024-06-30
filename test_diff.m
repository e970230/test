%% 測試diff評判差值過大情況

clc
clear
close all

x=1:1000;
y=0.005*x+0.005;

y_test=7*ones(1,100);

y(201:300)=y_test;
y(701:800)=y_test;

look_diff_y=diff(y);
positions = find(look_diff_y > 1);

D2=y(positions(1)-50:positions(1)-1);


plot(x,y)