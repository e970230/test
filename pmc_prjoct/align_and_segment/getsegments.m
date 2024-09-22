function [servo_time_M123,servo_current_M123,time_nidaq_M123,data_nidaq_M123] = getsegments(servo_time,servo_pos,servo_current,data_nidaq,time_nidaq,segments,dir)
%UNTITLED3 Summary of this function goes here
% 根據servo guide的位置訊號進行訊號切分
% 並分出每一段去程

%模型1: -13.18 ~ -123.84 mm  (F10000)
%模型2: -123.84 ~ -234.68 mm  (F10000)
%模型3: -456.01 ~ -566.84 mm  (F10000)

%模型1: -35.5705 ~ -137.4371 mm  (F16000)
%模型2: -137.4371 ~ -239.0371mm  (F16000)
%模型3: -442.7705 ~ -544.3705 mm  (F16000)


% input
%-----------------------------------------------------------
% servo_pos: servo guide的位置訊號; 維度=訊號長度*1
% servo_time: servo_pos的時間軸; 維度=訊號長度*1
% data_nidaq: 已與位置訊號對齊後的NIDAQ data; 維度=訊號長度*channel數
% time_nidaq: data_nidaq的時間軸; 維度=訊號長度*1
% segments: 切分後的訊號對應的起點與終點位置 (mm); 維度=模型數*2
% dir: 1:只選擇去程, 2:只選擇回程, 3:去回程都要


% output
%-----------------------------------------------------------
% servo_current_M123: 切分好的電流訊號; cell維度=去程段數*模型數; 矩陣維度=訊號長度*1
% servo_time_M123: 切分好的電流訊號所對應的時間軸; cell維度=去程段數*模型數; 矩陣維度=訊號長度*1
% data_nidaq_M123: 切分好的nidaq訊號; cell維度=去程段數*模型數; 矩陣維度=訊號長度*channel數
% time_nidaq_M123: 切分好的nidaq訊號所對應的時間軸; cell維度=去程段數*模型數; 矩陣維度=訊號長度*1



% last modification: 2024/08/30



%判斷去回程共有幾段
inx01=find((servo_pos<=segments(1,1)) & (servo_pos>=segments(1,2)));
inx02=find(diff(inx01)>1);
inx03=inx01(inx02);
N=length(inx03)+1;  %去回程總段數



inx4=cell(3,N);  %在指定位置範圍內, 去程與回程的索引; 維度=模型數*去回程總段數

for h=1:3
    inx1=find((servo_pos<=segments(h,1)) & (servo_pos>=segments(h,2)));
    inx2=find(diff(inx1)>1);
    inx3=inx1(inx2);

    for k=1:N
        if k==1
            inx4{h,k}=inx1(1):inx3(1);
        elseif k==N
            inx4{h,k}=inx1(inx2(end)+1):inx1(end);
        else
            inx4{h,k}=inx1(inx2(k-1)+1):inx3(k);
        end
    end
end



if dir==1  %只保留去程
   take_inx=1:2:(N-1);
elseif dir==2  %只保留回程
    take_inx=2:2:(N);
else   %去回程都要
    take_inx=1:1:(N);
end



%切分好的電流訊號與對應的時間軸
servo_time_M123=cell(length(take_inx),3);
servo_current_M123=cell(length(take_inx),3);

%切分好的nidaq訊號與對應的時間軸
time_nidaq_M123=cell(length(take_inx),3);
data_nidaq_M123=cell(length(take_inx),3);



for h=1:3

    for v=1:length(take_inx)
        segments_inx=inx4{h,take_inx(v)};  %在指定位置範圍內, 去程的索引

        segment_time=servo_time([segments_inx(1) segments_inx(end)],1);  %所切分區段的起/終點所對應的時間 (unit: sec)

        for w=1:2
            [~,segments_inx_servo(w)]=min(abs(servo_time-segment_time(w)));  %所切分位置對應於servo guide資料的index
            [~,segments_inx_nidaq(w)]=min(abs(time_nidaq-segment_time(w)));  %所切分位置對應於nidaq資料的index
        end

        inx_servo=segments_inx_servo(1):segments_inx_servo(2);

        servo_time_M123{v,h}=servo_time(inx_servo,1);
        servo_current_M123{v,h}=servo_current(inx_servo,1);


        inx_nidaq=segments_inx_nidaq(1):segments_inx_nidaq(2);
        time_nidaq_M123{v,h}=time_nidaq(inx_nidaq,1);
        data_nidaq_M123{v,h}=data_nidaq(inx_nidaq,:);


    end


end




end