%測試小波降噪練習

clc
clear
close all

%小波包分解
%離散小波變換和小波分解

%----------------------------------------------------
%創建模擬訊號
fs=200;
t=0:1/fs:2;

y=linspace(0,10,length(t));

noise=5*randn(1,length(y));   %量測雜訊 (zero-mean noise)
yn=y+noise; %加上量測雜訊的模擬量測值Y

figure(1)
plot(t,y,t,yn,'r-.')

%小波拆解
wname='db7';
[c,l] = wavedec(yn,3,wname);
% c是小波分解的層數
% 以這題為例:l是長3+2的向量，包含每個尺度的小波係數的長度，以及原始信號的長度
approx = appcoef(c,l,wname);
%appcoef是可以獲取小波分解的近似係數，以及低頻信息(訊號的大致走向
[cd1,cd2,cd3] = detcoef(c,l,[1 2 3]);
%detcoef可以獲得小波分解的近似係數，以及高頻信息(訊號的細節
figure(2)
subplot(4,1,1)
plot(approx)
title('Approximation Coefficients')
subplot(4,1,2)
plot(cd3)
title('Level 3 Detail Coefficients')
subplot(4,1,3)
plot(cd2)
title('Level 2 Detail Coefficients')
subplot(4,1,4)
plot(cd1)
title('Level 1 Detail Coefficients')


%小波重構
c_approx=c(1:l(1));   %近似項的係數
c_details=c(l(1)+1:end);   %細節項的係數

c_details0=c_details;

threshold=1;   %篩選門檻, 0~1, 低於門檻的係數值(normalized)皆令其為0
c_details0((abs(c_details)./max(abs(c_details)))<=threshold)=0;   %細節項中, abs(係數)<threshold的直接令其為0
%此作法即省略掉不重要的細節處, 以達降噪目的

c=[c_approx,c_details0];   %處理完畢後的係數項
y_wavelet = waverec(c,l,wname);   %小波重構


%使用moving average降噪
y_movavg = movavg( yn,10 );

figure(3)
plot(t,y,'color','b','linewidth',2)
hold on
plot(t,yn,'color',[126/255,126/255,126/255],'linestyle','-.')
plot(t,y_wavelet,'color','r','linewidth',2)
plot(t,y_movavg,'linewidth',2,'color',[237/255,177/255,32/255])
hold off
legend('no noise','with noise','filtered (wavelet)','filtered (moving average)')

%比較不同篩選門檻 (不同小波降噪效力)的小波重構結果


threshold=[1,0.3];   %篩選門檻, 0~1, 低於門檻的係數值(normalized)皆令其為0

c_all=[];
y_wavelet_all=[];


for q=1:length(threshold)
    
    c_details_p=c_details;
    c_details_p((abs(c_details)./max(abs(c_details)))<threshold(q))=0;
    c_all=[c_all;[c_approx,c_details_p]];   %處理完畢後的係數項
    y_wavelet_all = [y_wavelet_all;waverec(c_all(q,:),l,wname)];   %小波重構
end


figure(4)
plot(t,y,'color','b','linewidth',2)
hold on
plot(t,yn,'color','k','linestyle','-.')
plot(t,y_wavelet_all(2,:),'color','r','linewidth',1)
plot(t,y_wavelet_all(1,:),'linewidth',2,'color',[237/255,177/255,32/255])
hold off
legend('no noise','with noise','threshold=0.3','threshold=1')
