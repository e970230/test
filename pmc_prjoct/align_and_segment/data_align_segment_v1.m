% Step 1.
% 以servo guide的位置訊號為基準, 調整轉速計訊號, 
% 使調整後的轉速計訊號能與servo guide的位置訊號對齊
% nidaq訊號皆配合著轉速計訊號的改動, 一併與servo guide的位置訊號對齊

% Step 2.
% 針對已與servo guide的位置訊號對齊的nidaq訊號, 
% 根據servo guide的位置訊號進行訊號切分, 並分出每一段去程



% last modification: 2024/09/12



clear
clc
close all

fs=12800;  % NI採樣頻率
fs_servo=4000;  % servo guide採樣頻率
% ch1: 頭端加速規
% ch2: 螺帽加速規
% ch3: 尾端加速規
% ch4: 尾端位移計
% ch5: 頭端位移計
% ch6: 中段雷射位移計
% ch7: tachometer
% servo: servo guide
%  第1行: index
%  第2行: 時間
%  第3行: 位置
%  第4行: 速度 
%  第5行: 電流百分比


%% 讀取資料

loc='C:\Users\user\Desktop\新增資料夾\dlp_0_F10000';  %資料存放位置

savefile='dataset_dlp_0_F10000.mat';  %存檔檔名



for h=1:2  %依序讀取第1~100趟, 第101~200趟, ...

    %NI DAQ資料
    Ch1 = csvread(fullfile(loc,['dlp_0_f10000_Ch1_' num2str(h) '.csv']),0,1);
    Ch2 = csvread(fullfile(loc,['dlp_0_f10000_Ch2_' num2str(h) '.csv']),0,1);
    Ch3 = csvread(fullfile(loc,['dlp_0_f10000_Ch3_' num2str(h) '.csv']),0,1);
    Ch4 = csvread(fullfile(loc,['dlp_0_f10000_Ch4_' num2str(h) '.csv']),0,1);
    Ch5 = csvread(fullfile(loc,['dlp_0_f10000_Ch5_' num2str(h) '.csv']),0,1);
    Ch6 = csvread(fullfile(loc,['dlp_0_f10000_Ch6_' num2str(h) '.csv']),0,1);
    Ch7 = csvread(fullfile(loc,['dlp_0_f10000_Ch7_' num2str(h) '.csv']),0,1);


    % servo guide資料
    servo_data= csvread(fullfile(loc,['dlp_0_f10000_servo_' num2str(h) '.csv']),0,0); %servo guide
    servo_time=servo_data(:,1);  % time vector (unit: sec)
    servo_pos=servo_data(:,2);  % position
    servo_current=servo_data(:,4);  % 電流訊號 (%)
    

    %執行對齊
    data2align=[Ch1,Ch2,Ch3,Ch4,Ch5,Ch6,Ch7];   %欲與位置訊號對齊的資料; 維度=訊號長度*channel數

    tacho=data2align(:,7);  %轉速計訊號

    [data_aligned,time_aligned] = getalign(tacho,servo_pos,fs,fs_servo,data2align);



    %檢查對齊
    figure(h)
    [hAx,hLine1,hLine2] = plotyy(time_aligned,data_aligned(:,7),servo_time,servo_pos);
    xlabel('Time (sec)')
    ylabel(hAx(1),"tachometer (Voltage)") % left y-axis 
    ylabel(hAx(2),"position (mm)") % right y-axis
    grid on
    legend('tacho','position (servo guide)')



    %根據預先規劃的切分位置, 將位置訊號切分

    %模型1: -13.18 ~ -123.84 mm  (F10000)
    %模型2: -123.84 ~ -234.68 mm  (F10000)
    %模型3: -456.01 ~ -566.84 mm  (F10000)

    %模型1: -35.5705 ~ -137.4371 mm  (F16000)
    %模型2: -137.4371 ~ -239.0371mm  (F16000)
    %模型3: -442.7705 ~ -544.3705 mm  (F16000)


    % F10000
    segments=[-13.18,-123.84; -123.84,-234.68; -456.01,-566.84];  %row1: 模型1的行程起點與終點位置 (mm)

    % F16000
    % segments=[-35.5705,-137.4371; -137.4371,-239.0371; -442.7705,-544.3705];  %row1: 模型1的行程起點與終點位置 (mm)




    dir=1;  % dir: 1:只選擇去程, 2:只選擇回程, 3:去回程都要
    [servo_time_cut,servo_current_cut,time_nidaq_cut,data_nidaq_cut] = getsegments(servo_time,servo_pos,servo_current,data_aligned,time_aligned,segments,dir);
    % servo_current_cut: 切分好的電流訊號; cell維度=去程段數*模型數; 矩陣維度=訊號長度*1
    % servo_time_cut: 切分好的電流訊號所對應的時間軸; cell維度=去程段數*模型數; 矩陣維度=訊號長度*1
    % data_nidaq_cut: 切分好的nidaq訊號; cell維度=去程段數*模型數; 矩陣維度=訊號長度*channel數
    % time_nidaq_cut: 切分好的nidaq訊號所對應的時間軸; cell維度=去程段數*模型數; 矩陣維度=訊號長度*1



    


    %檢查訊號切分結果
    % 檢查nidaq訊號(螺帽振動)切分結果 (模型3)
    figure(10+h)
    hold on
    plot(time_aligned,data_aligned(:,2))

    for k=1:size(time_nidaq_cut,1)
        plot(time_nidaq_cut{k,3},data_nidaq_cut{k,3}(:,2),'r.')
    end
    hold off
    xlabel('Time (sec)')
    ylabel('Vibration (g)')
    grid on



    if h==1
        ALL_servo_current=servo_current_cut;
        ALL_servo_time=servo_time_cut;
        ALL_data_nidaq=data_nidaq_cut;
        ALL_time_nidaq=time_nidaq_cut;
    else
        ALL_servo_current=[ALL_servo_current ; servo_current_cut];
        ALL_servo_time=[ALL_servo_time ; servo_time_cut];
        ALL_data_nidaq=[ALL_data_nidaq ; data_nidaq_cut];
        ALL_time_nidaq=[ALL_time_nidaq ; time_nidaq_cut];
    end




end
%%

save(fullfile(loc,savefile),'ALL_servo_time','ALL_servo_current','ALL_time_nidaq','ALL_data_nidaq','fs_servo','fs');


