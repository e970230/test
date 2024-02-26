function [ filtered_ach_force ] = movavg( ach_force,movavg_size )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

windowSize = movavg_size;  %中央moving average的窗函數長度  (單位:點數)
y1=filter(ones(1,fix(movavg_size/2)+1)/windowSize,1,ach_force);
y2=filter(ones(1,fix(movavg_size/2)+1)/windowSize,1,fliplr(ach_force));
filtered_ach_force=y1+fliplr(y2)-(1/movavg_size)*ach_force;

end

