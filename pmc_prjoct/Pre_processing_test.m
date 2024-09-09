clc
clear
close all

%此為113-PMC螺桿系統動態監測分析與狀態估測模組開發的前程式開發檔案

% 指定資料夾路徑
folderPath = 'C:\Users\林坪橙\Documents\GitHub\test\pmc_prjoct'; % 替換成你的資料夾路徑

% 獲取資料夾中所有的 CSV 檔案
matFiles = dir(fullfile(folderPath, '*.mat'));

% 提取檔名
fileNames = {matFiles.name};

for df=1:size(fileNames,2)
    init_data(df,:) = load(fileNames{df}).data;
end



data_title = {'近馬達端軸承座加速規' '螺桿螺帽加速規' '尾端軸承座加速規' '直軌尾端(遠端馬達端)電容式位移計' '直軌頭端(近馬達端)電容式位移計' '直軌中段雷射位移計' '轉速計' '伺服器馬達電流鉤錶'};
data_xlabel = {'Time(sec)'};
data_ylabel = {'g' 'g' 'g' '\mu_m' '\mu_m' 'mm' 'V' 'A'};


% hold on
% for i=1:size(init_data,1)
%     figure(i)
%     x = 1:size(init_data{i,6},1);
%     plot(x,init_data{i,6})
%     title(data_title{6})
%     xlabel(data_xlabel)
%     ylabel(data_ylabel{6})
% end

% hold off
%------------------------------------處理負訊號嘗試-------------------------
% find_num = 3;
% 
% prcoss_data = init_data{find_num,6};
% for i=1:size(prcoss_data,1)
%     if prcoss_data(i)<=6.285
%         prcoss_data(i)=prcoss_data(i-1);
%     end
% end
% figure(1)
% x = 1:size(prcoss_data,1);
% plot(x,prcoss_data)
% title('將負訊號排除掉後訊號')
% xlabel('Time(sec)')
% ylabel('mm')
% 
% figure(2)
% x = 1:size(init_data{find_num,6},1);
% plot(x,init_data{find_num,6})
% title('原始訊號')
% xlabel('Time(sec)')
% ylabel('mm')
%--------------------------------------------------------------------------


find_num = 3;

prcoss_data = init_data{find_num,6};

figure(2)
x = 1:size(init_data{find_num,6},1);
plot(x,init_data{find_num,6})
title('原始訊號')
xlabel('Time(sec)')
ylabel('mm')

%-----------------------------------自行開發處------------------------------

normalized_prcoss_data = normalize(prcoss_data, 'range');             %將數據正規化到 [0, 1] 範圍內
postion_prcoss_data = find(normalized_prcoss_data == 1);             %將正規化後數字超過0.95的位置抓出來
        
diff_prcoss_data = diff(postion_prcoss_data);           %得出差值
ans_01 =[1;find(diff_prcoss_data~=1);size(postion_prcoss_data,1)];                     %把前後差值不是1抓出來，想找出前後不連續的位置
correct_point = find(diff(ans_01)~=1);
look_for_point = ans_01(correct_point)+1;



diff_ans_01 = diff(ans_01);
rp = 0;
for i=1:size(diff_ans_01,1)-1

    if diff_ans_01(i+rp)==1
        if diff_ans_01(i+rp+1)==1
            rp = rp + 1;
        end
        answer(i,:) = [ans_01(i+1+rp)+1 ans_01(i+2+rp)];
        rp = rp + 1;
        
        
    else
        answer(i,:) = [ans_01(i+rp)+1 ans_01(i+1+rp)];
    end
    
    look = rp+i;

    if look==size(diff_ans_01,1)
        answer(1,1)=1;
        break
    end
end


final_data = postion_prcoss_data(answer)


%--------------------------------------------------------------------------

order = 3;      %濾波器的階數，此值須小於移動窗口的長度
framelen = 193;  %移動窗口的長度(必須為奇數

sg_prcoss_data = sgolayfilt(prcoss_data,order,framelen);
[output]=diff_signal(prcoss_data,200,0.1);



figure(3)
x = 1:size(sg_prcoss_data,1);
plot(x,sg_prcoss_data)
title('sg濾波器處理後訊號')
xlabel('Time(sec)')
ylabel('mm')
figure(2)
x = 1:size(init_data{find_num,6},1);
plot(x,init_data{find_num,6})




