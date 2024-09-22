
clc
clear
close all


%讀取檔案

load("dataset_dlp_10_F16000.mat")


data = ALL_servo_current(:,3);

%%

%頻率解析度
fs=4000;

%進給速度
F=10000;

%一倍頻
spindle_fs=F/(12*60);   

%要找的轉頻目標
spendle_fs_list=spindle_fs*(4:4:8);

%搜尋範圍
tolerance=5;


feature=[];

for i=1:length(data)
    
    extract_value=[];


    disp([num2str(i),' / ',num2str(length(data))])
    
    
    %依序提取出每一筆資料
    fs_signal=data{i}(:,1)';
    % fs signal維度為 [1*量測次數]


    % 計算均方根 (RMS)
    rms_value = rms(fs_signal);
    
    % 計算峭度 (Kurtosis)
    kurtosis_value = kurtosis(fs_signal);
    
    % 計算偏態 (Skewness)
    skewness_value = skewness(fs_signal);
    
    % 計算峰對峰值 (Peak-to-Peak)
    peak_to_peak_value = peak2peak(fs_signal);
    
    % 計算標準差 (Standard Deviation)
    std_value = std(fs_signal);


    extract_value = [rms_value,kurtosis_value,skewness_value,peak_to_peak_value,std_value];


    %fft
    windowl=hann(length(fs_signal));
    singal_win=fs_signal.*windowl';
    Y2=fft(singal_win);
    L=length(singal_win);
    P2_win=abs(Y2/L);
    P1_win=P2_win(1:L/2+1);
    P1_win(2:end-1)=2*P1_win(2:end-1);
    Peak=P1_win*2;
    f=fs*(0:(L/2))/L;
    
    
    %find spendle rotation vib
    for j=spendle_fs_list
        
        % find the range
        ind=((f>(j-tolerance)) & (f<(j+tolerance)));
    
        %find the biggest peak in the range
        value=max(Peak(ind));
        
        extract_value=[extract_value,value];
    end
    
    feature=[feature;extract_value];
   
end