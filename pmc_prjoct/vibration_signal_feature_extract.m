
clc
clear
close all


%讀取檔案

load("dataset_dlp_10_F16000.mat")


%%

%數字控制模型
data = ALL_data_nidaq(:,2);

%頻率解析度
fs=12800;

%進給速度
F=16000;

%一倍頻
spindle_fs=F/(12*60);   

%要找的轉頻目標
spendle_fs_list=spindle_fs*(4:4:32);

%搜尋範圍
tolerance=5;

% BD=band energy
BD_delta=30;

BD_start_fs=18*spindle_fs;
BD_end_fs=200*spindle_fs;

BD_list=BD_start_fs:BD_delta:BD_end_fs;

BD_list=BD_list(1:end-1);


total_ch_feature={};

% 模型1用ch5
% 模型2用ch6
% 模型3用ch4

%% ------------output
for i=1:length(data)

    output_signal=data{i}(:,5)'; %這邊控制(多少趟去程*第幾個ch)
    
    Fianl_labels(i,:) = median(output_signal);

end



for i=1:length(data)

    

    disp([num2str(i),' / ',num2str(length(data))])
    
    % 1:3代表我要找ch1~3的訊號
    for ch = 1:3
        
        extract_value=[];
        feature = [];
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
       
       
        
        feature=[feature;extract_value];

        total_ch_feature{i,ch} = feature ;

    end
end

for tcf = 1:size(total_ch_feature,1)

    Final_feature(tcf,:) = [total_ch_feature{tcf,1} total_ch_feature{tcf,2} total_ch_feature{tcf,3}];
    
    disp([num2str(tcf),' / ',num2str(length(total_ch_feature))])
end

