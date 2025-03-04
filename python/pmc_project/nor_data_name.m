%正規化後資料

clc
clear
close all


F = 10000;


F_total = [10000 16000];    %直軌進給速度
out_num_total = [0 10 20 30 40 50];     %直軌偏移量

%想要尋找的轉頻目標(幾倍頻到幾倍頻之類的)
% example: 4:4:32 代表想從4倍頻找到32倍頻且每4倍頻為一個單位
spindle_fs_target = 4:4:32;

%想要尋找的電流訊號轉頻目標
current_data_spindle_fs_target = 4:4:8;


%頻率開始和結束
BD_spindle_start = 18;
BD_spindle_end = 100;

spindle_fs=F/(12*60);   

%搜尋範圍
tolerance=5;

spendle_fs_list=spindle_fs*(spindle_fs_target);

% % BD=band energy
% if F==10000
%     BD_delta=20;
% elseif F==16000
%     BD_delta=20;
% end

BD_delta_all = 0.125; %新版間距已倍頻角度觀看

BD_start_fs=BD_spindle_start*spindle_fs;
BD_end_fs=BD_spindle_end*spindle_fs;

BD_list=BD_start_fs:BD_delta_all*spindle_fs:BD_end_fs;





for ch=1:3
    if ch==1
        Labels_name = {
            [' ch. ' num2str(ch) '均方根值 (RMS)'], ...
            [' ch. ' num2str(ch) '峭度 (Kurtosis)'], ...
            [' ch. ' num2str(ch) '偏態 (Skewness)'], ...
            [' ch. ' num2str(ch) '標準差 (Standard Deviation)'], ...
            [' ch. ' num2str(ch) '峰值因子 (Crest Factor)']
        };
    else
        Labels_name(end+1:end+5) = {
            [' ch. ' num2str(ch) '均方根值 (RMS)'], ...
            [' ch. ' num2str(ch) '峭度 (Kurtosis)'], ...
            [' ch. ' num2str(ch) '偏態 (Skewness)'], ...
            [' ch. ' num2str(ch) '標準差 (Standard Deviation)'], ...
            [' ch. ' num2str(ch) '峰值因子 (Crest Factor)']
        };
        
    end
    
    for i = 1:size(spindle_fs_target,2)
        Labels_name_site = ['ch.' num2str(ch) ' ' num2str(spindle_fs_target(i)) ' 倍階次域 '];
        Labels_name{end+1} = Labels_name_site;
    end
    
    for i = 1:size(spindle_fs_target,2)
        Labels_name_site = ['ch.' num2str(ch) ' ' num2str(spindle_fs_target(i)) ' 倍階次域能量占比 '];
        Labels_name{end+1} = Labels_name_site;
    end
    
    for i = 1:size(BD_list,2)-1
        Labels_name_site = ['ch.' num2str(ch) ' ' num2str(BD_list(i)) ' HZ到 ' num2str(BD_list(i+1)) ' HZ的頻譜帶能量'];
        Labels_name{end+1} = Labels_name_site;
    end
    
    for i = 1:size(BD_list,2)-1
        Labels_name_site = ['ch.' num2str(ch) ' ' num2str(BD_list(i)) ' HZ到 ' num2str(BD_list(i+1)) ' HZ的頻譜帶能量占有比例'];
        Labels_name{end+1} = Labels_name_site;
    end

end

Labels_name(end+1:end+5) = {
    ['電流訊號均方根值 (RMS)'], ...
    ['電流訊號峭度 (Kurtosis)'], ...
    ['電流訊號偏態 (Skewness)'], ...
    ['電流訊號標準差 (Standard Deviation)'], ...
    ['電流訊號峰值因子 (Crest Factor)']
};

for i = 1:size(current_data_spindle_fs_target,2)
    Labels_name_site = [' 電流訊號轉換諧波之 ' num2str(current_data_spindle_fs_target(i)) ' 倍階次域 ' ];
    Labels_name{end+1} = Labels_name_site;
end

for i = 1:size(current_data_spindle_fs_target,2)
    Labels_name_site = [' 電流訊號轉換諧波之 ' num2str(current_data_spindle_fs_target(i)) ' 倍階次域能量占有比例 ' ];
    Labels_name{end+1} = Labels_name_site;
end