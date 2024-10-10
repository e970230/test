%前置要求若在讀取檔案時沒有(fs NI採樣頻率、fs_servo servo guide採樣頻率)必須自行設定

%此為整合版特徵提取程式，將振動訊號和電流訊號以及對應的labels提取出來，並儲存成相對應的檔案

clc
clear
close all


%讀取檔案

load("dataset_dlp_10_F10000.mat")



%% 需自行手動設定部分

%偏移量
out_num = 10;

%進給速度
F=10000;

%搜尋範圍
tolerance=5;

%一倍頻
spindle_fs=F/(12*60);   


BD_delta_F10000 = 20;
BD_delta_F16000 = 30;


for k = 1:3 %有3個模型，故這裡迴圈設定1~3

    %數字控制模型
    models_num = k;
    
    data = ALL_data_nidaq(:,models_num);
    
    %頻率解析度
    % fs=12800;
    
    
    %要找的轉頻目標
    spendle_fs_list=spindle_fs*(4:4:32);
    
    
    
    
    % BD=band energy
    if F==10000
        BD_delta=BD_delta_F10000;
    elseif F==16000
        BD_delta=BD_delta_F16000;
    end
    
    
    BD_start_fs=18*spindle_fs;
    BD_end_fs=200*spindle_fs;
    
    BD_list=BD_start_fs:BD_delta:BD_end_fs;
    
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
    
    
    %% ------------output
    for i=1:length(data)
    
        output_signal=data{i}(:,models_labels)'; %這邊控制(多少趟去程*第幾個ch)
        
        Fianl_labels(i,:) = median(output_signal);
    
    end
    
    
    
    for i=1:length(data)
    
        
    
        disp([num2str(i),' / ',num2str(length(data))])
        
        % 1:3代表我要找ch1~3的訊號
        for ch = 1:3
            
            extract_value=[];
            vibration_feature = [];
            %依序提取出每一筆資料
            fs_signal=data{i}(:,ch)'; %這邊控制(多少趟去程*第幾個ch)
        
        
            %time doman
            rms_value = rms(fs_signal);
            kurtosis_value = kurtosis(fs_signal);
            skewness_value = skewness(fs_signal);
            std_value = std(fs_signal);
            crest_factor = max(abs(fs_signal)) / rms(fs_signal);
        
            extract_value = [rms_value,kurtosis_value,skewness_value,std_value,crest_factor];
        
        
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
                end_value=((f>((j+BD_delta)-tolerance)) & (f<((j+BD_delta)+tolerance)));
            
                %find the biggest peak in the range
                end_ind=max(Peak(end_value));
        
                end_ind=find(Peak==end_ind);
        
                BD_value = sum(Peak(start_ind:end_ind));
        
                BD_feature=[BD_feature,BD_value];
            end
        
        
            BD_per=BD_feature/sum(BD_feature);
            
            extract_value=[extract_value,BD_feature,BD_per];
           
           
            
            vibration_feature=[vibration_feature;extract_value];
    
            total_ch_vibration_feature{i,ch} = vibration_feature ;
    
        end
    end
    
    for tcf = 1:size(total_ch_vibration_feature,1)
    
        Final_vibration_feature(tcf,:) = [total_ch_vibration_feature{tcf,1} total_ch_vibration_feature{tcf,2} total_ch_vibration_feature{tcf,3}];
        
        disp([num2str(tcf),' / ',num2str(length(total_ch_vibration_feature))])
    end
    
    %% 電流訊號
    
    %要找的轉頻目標
    spendle_fs_list=spindle_fs*(4:4:8);
    
    
    current_data = ALL_servo_current(:,models_num);
    
    
    Final_current_feature=[];
    
    for i=1:length(current_data)
        
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
        f=fs_servo*(0:(L/2))/L;
        
        
        %find spendle rotation vib
        for j=spendle_fs_list
            
            % find the range
            ind=((f>(j-tolerance)) & (f<(j+tolerance)));
        
            %find the biggest peak in the range
            value=max(Peak(ind));
            
            extract_value=[extract_value,value];
        end
        
        Final_current_feature=[Final_current_feature;extract_value];
       
    end
    
    %% 最終檔案輸出
    
    % loc='D:\pmc計畫資料存放';  %資料存放位置
    
    save_input_file=['input_data_models_' num2str(models_num)  '_dlp_' num2str(out_num) '_F_' num2str(F) '.mat'];  %存檔檔名
    
    save_output_file=['output_data_models_' num2str(models_num)  '_dlp_' num2str(out_num) '_F_' num2str(F) '.mat'];  %存檔檔名
    

    Final_input_data = [Final_vibration_feature Final_current_feature];
    
    save(fullfile(save_input_file),'Final_input_data');

    save(fullfile(save_output_file),'Fianl_labels');

end
