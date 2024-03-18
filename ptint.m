clc
clear
close all


A=load("5_10_02.mat","Population_answer");
A=A.Population_answer;
AA=A(1,4,1:size(A,3));

for y=1:size(A,3)
    plot_y(y)=AA(1,1,y);
end

plot(1:size(A,3),plot_y)