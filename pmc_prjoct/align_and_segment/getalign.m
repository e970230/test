function [data_aligned,time_aligned] = getalign(tacho,servo_pos,fs,fs_servo,data2align)
%UNTITLED5 Summary of this function goes here

%以servo guide的位置訊號為基準, 調整轉速計訊號, 
%使調整後的轉速計訊號能與servo guide的位置訊號對齊
%振動訊號及位移計訊號皆配合著轉速計訊號的改動, 一併與servo guide的位置訊號對齊

% input
%-------------------------------------------------
% tacho: 轉速計訊號; 維度=訊號長度*1
% fs: tacho的採樣頻率(Hz)
% servo_pos: servo guide的位置訊號; 維度=訊號長度*1
% fs_servo: servo guide的採樣頻率(Hz)
% data2align: 欲與位置訊號對齊的資料; 維度=訊號長度*channel數

% output
%-------------------------------------------------
% data_aligned: 與位置訊號對齊後的data2align; 維度=訊號長度*channel數
% time_aligned: data_aligned的時間軸; 維度=訊號長度*1



% last modification: 2024/08/30




time_Ch7=((0:1:length(tacho)-1)/fs)';
servo_time=((0:1:length(servo_pos)-1)/fs_servo)';

Ch7_re01=normalize(tacho,'range');  %將轉速計訊號正規化至[0,1]

%找出轉速計第一個脈衝的時間點與位置訊號甫移動至0.01mm時的時間點, 計算二時間點的時間差
time_gap=time_Ch7(min(find(Ch7_re01>0.9)))-servo_time(min(find(servo_pos<=-0.01)));

%將此時間差換算為點數(因為是要調整轉速計訊號, 故使用轉速計的採樣頻率)
shift_gap=abs(fix(time_gap*fs));



data_aligned=data2align(shift_gap:end,:);
time_aligned=((0:1:size(data_aligned,1)-1)/fs)';


end