%測試將頻率正規化後程式
%此次測試目標為想找出無視進給速度可以區分偏移量之特徵
%前置要求若在讀取檔案時沒有(fs NI採樣頻率、fs_servo servo guide採樣頻率)必須自行設定

clc
clear
close all

fuck_error_time = 0;

F_total = [10000 16000];    %直軌進給速度
out_num_total = [0 10 20 30 40 50];     %直軌偏移量


%想要尋找的轉頻目標(幾倍頻到幾倍頻之類的)
% example: 4:4:32 代表想從4倍頻找到32倍頻且每4倍頻為一個單位
spindle_fs_target = 4:4:32;

%想要尋找的電流訊號轉頻目標
current_data_spindle_fs_target = 4:4:8;


%頻率間隔
BD_delta_F10000 = 20;
BD_delta_F16000 = 20;
BD_delta_all = 0.125; %新版間距已倍頻角度觀看

%頻率開始和結束
BD_spindle_start = 18;
BD_spindle_end = 100;


%搜尋範圍
tolerance=2;

%2024/10/20 由於實驗498趟出現問題所以接下來所有資料均採用500-4 = 496筆資料
%2024/11/27 由於某筆資料裡第98趟出問題，所以選用497趟資料，並且第因為少了第96趟資料，所以總資料依樣是496趟
data_num = 1:497; %採用的趟數

for it_one = 1:size(out_num_total,2)
    
    %偏移量
    out_num = out_num_total(it_one);
    for it_two = 1:size(F_total,2)
        
        clear Final_vibration_feature
        clear end_ind
        clear Fianl_labels
        clear Final_input_data

        
        %進給速度
        F=F_total(it_two);
        

        
        %一倍頻
        spindle_fs=F/(12*60);   
        
        
        
        %讀取檔案
        
        load(['dataset_Test' num2str(it_one) '_F' num2str(F) '.mat'])
        
        
        for k = 1:3 %有3個模型，故這裡迴圈設定1~3
        
            %數字控制模型
            models_num = k;
            
            data = ALL_data_nidaq(data_num,models_num);
            
            
            %要找的轉頻目標
            spendle_fs_list=spindle_fs*(spindle_fs_target);
            
            
            
            
            
            
            BD_start_fs=BD_spindle_start*spindle_fs;
            BD_end_fs=BD_spindle_end*spindle_fs;
            
            
            BD_list=BD_start_fs:BD_delta_all*spindle_fs:BD_end_fs;
            
            BD_list=BD_list(1:end-1);
            
           


            total_ch_vibration_feature={};
            
            % 模型1用ch5
            % 模型2用ch6
            % 模型3用ch4
            
            if models_num==1
                models_labels = 5;
            elseif models_num==2
                models_labels = 6;
            elseif models_num==3
                models_labels = 4;
            end
            
            fuck_error_time = 0;
            %% ------------output
            for i=1:length(data)
                
                %----------異常資料跳過----------
                if i==98
                    fuck_error_time = fuck_error_time+1;
                    continue; % 跳過這次迴圈的處理
                end
                %-------------------------------
                output_signal=data{i}(data_num,models_labels)'; %這邊控制(多少趟去程*第幾個ch)
                
                Fianl_labels(i-fuck_error_time,:) = median(output_signal);
            
            end
            
            fuck_error_time = 0;
            
            for i=1:length(data)
            
                %----------異常資料跳過----------
                if i==98
                    fuck_error_time = fuck_error_time+1;
                    continue; % 跳過這次迴圈的處理
                end
                %-------------------------------
            
                disp([num2str(i),' / ',num2str(length(data))])
                
                % 1:3代表我要找ch1~3的訊號
                for ch = 1:3
                    
                    extract_value=[];
                    vibration_feature = [];
                    time_value = [];
                    fft_value = [];
                    %依序提取出每一筆資料
                    fs_signal=data{i}(:,ch)'; %這邊控制(多少趟去程*第幾個ch)
                
                
                    %time doman
                    rms_value = rms(fs_signal);
                    kurtosis_value = kurtosis(fs_signal);
                    skewness_value = skewness(fs_signal);
                    std_value = std(fs_signal);
                    crest_factor = max(abs(fs_signal)) / rms(fs_signal);
                
                    time_value = [rms_value,kurtosis_value,skewness_value,std_value,crest_factor];
                
                
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
                        
                        if all(ind == 0)
                            
                            keyboard; %進入交互模式，輸入 return 後繼續執行
                        end

                        %find the biggest peak in the range
                        value=max(Peak(ind));
                        

                        fft_value=[fft_value,value];
                    end
                    %---------------找到一倍頻的位置及能量
                    % find the range
                    ind_frist=((f>(spindle_fs-tolerance)) & (f<(spindle_fs+tolerance)));
                
                    %find the biggest peak in the range
                    value_frist=max(Peak(ind_frist));
                    %------------------------------------
                    
                    peak_vavlue_nor = fft_value/value_frist;           %4倍頻~32倍頻
                    peak_vavlue_nor_2 = fft_value/sum(fft_value);
                    
                    
                    BD_feature=[];
                
                    %find band energy
                    % j = frequence
                    for j=BD_list
                
                        % find the range
                        start_value=((f>(j-tolerance)) & (f<(j+tolerance)));
                    
                        %find the biggest peak in the range
                        start_ind=max(Peak(start_value));
                
                        start_ind=find(Peak==start_ind);
                
                        % find the range
                        end_value=((f>((j+BD_delta_all*spindle_fs)-tolerance)) & (f<((j+BD_delta_all*spindle_fs)+tolerance)));
                        

                        %find the biggest peak in the range
                        
                        end_ind=max(Peak(end_value));
                
                        end_ind=find(Peak==end_ind);
                
                        BD_value = mean(Peak(start_ind:end_ind));
                
                        BD_feature=[BD_feature,BD_value];
                    end
                
                
                    BD_per=BD_feature/sum(BD_feature);      %頻帶能量占比
                    
                    extract_value=[time_value,peak_vavlue_nor,peak_vavlue_nor_2,BD_feature,BD_per];   %輸出數據順序
                   
                   
                    
                    vibration_feature=[vibration_feature;extract_value];
                    
                    total_ch_vibration_feature{i-fuck_error_time,ch} = vibration_feature ;
            
                end
            end
            
            for tcf = 1:size(total_ch_vibration_feature,1)
            
                Final_vibration_feature(tcf,:) = [total_ch_vibration_feature{tcf,1} total_ch_vibration_feature{tcf,2} total_ch_vibration_feature{tcf,3}];
                
                disp([num2str(tcf),' / ',num2str(length(total_ch_vibration_feature))])
            end
            %% 電流訊號
            
            %要找的轉頻目標
            spendle_fs_list=spindle_fs*(current_data_spindle_fs_target);
            
            
            current_data = ALL_servo_current(data_num,models_num);
            
            
            Final_current_feature=[];
            
            
            fuck_error_time = 0;%重製
            
            for i=1:length(current_data)
                
                %----------異常資料跳過----------
                if i==98
                    fuck_error_time = fuck_error_time+1;
                    continue; % 跳過這次迴圈的處理
                end
                %-------------------------------

                extract_value=[];
            
            
                disp([num2str(i),' / ',num2str(length(current_data))])
                
                
                %依序提取出每一筆資料
                fs_signal=current_data{i}(:,1)';
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
            
            
                current_time_value = [rms_value,kurtosis_value,skewness_value,peak_to_peak_value,std_value];
            
            
                %fft
                windowl=hann(length(fs_signal));
                singal_win=fs_signal.*windowl';
                Y2=fft(singal_win);
                L=length(singal_win);
                P2_win=abs(Y2/L);
                P1_win=P2_win(1:L/2+1);
                P1_win(2:end-1)=2*P1_win(2:end-1);
                Peak=P1_win*2;
                f=fs_servo*(0:(L/2))/L;
                
                
                %find spendle rotation vib
                for j=spendle_fs_list
                    
                    % find the range
                    ind=((f>(j-tolerance)) & (f<(j+tolerance)));
                
                    %find the biggest peak in the range
                    value=max(Peak(ind));
                    
                    extract_value=[extract_value,value];
                end

                %---------------找到一倍頻的位置及能量
                % find the range
                ind_frist=((f>(spindle_fs-tolerance)) & (f<(spindle_fs+tolerance)));
            
                %find the biggest peak in the range
                value_frist=max(Peak(ind_frist));
                %------------------------------------
                
                peak_vavlue_nor = extract_value/value_frist;           %4倍頻~32倍頻
                peak_vavlue_nor_2 = extract_value/sum(extract_value);
                    
                
                Final_current_feature=[Final_current_feature;[current_time_value peak_vavlue_nor peak_vavlue_nor_2]];
               
            end

            save_input_file=['input_data_models_' num2str(models_num)  '_dlp_' num2str(out_num) '_F_' num2str(F) '_nor.mat'];  %存檔檔名
            
            save_output_file=['output_data_models_' num2str(models_num)  '_dlp_' num2str(out_num) '_F_' num2str(F) '_nor.mat'];  %存檔檔名
            
        
            Final_input_data = [Final_vibration_feature Final_current_feature];
            
            save(fullfile(save_input_file),'Final_input_data');
        
            save(fullfile(save_output_file),'Fianl_labels');
        end
    end
end
