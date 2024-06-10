clc
clear
close all

load AIDC_data_no4_T5.mat

x=1:length(T5);


order = 3;      %濾波器的階數，此值須小於移動窗口的長度
framelen = 93;  %移動窗口的長度(必須為奇數

y = sgolayfilt(T5,order,framelen);


figure(1)
hold on 
plot(x,T5,"Color","B")
plot(x,y,"Color","R")
hold off
legend('原始訊號','降噪後訊號')
