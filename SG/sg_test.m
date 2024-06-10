clc
clear
close all

load AIDC_data_no4_T5.mat

x=1:length(T5);


order = 3;
framelen = 93;

y = sgolayfilt(T5,order,framelen);


figure(1)
hold on 
plot(x,T5,"Color","R")
plot(x,y,"Color","G")
hold off

