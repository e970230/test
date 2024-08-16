function [output]=diff_signal(signal,find_num,threshold)

% input--------------------------------------------------------------------
% signal:需要處理的訊號
% find_num:需要擷取訊號隻數量(若想取的訊號長度為50則此地方填50)
% threshold:尋找想排除的訊號之門檻值(當前後訊號差距超過此值時
%                                  將此變化之訊號位置註記下來
% output-------------------------------------------------------------------
% output:處理完的訊號(訊號*起伏數量)，亦即若今天訊號起伏處為三處且想取得這三處
%        前30個資料點之對應訊號，則此輸出矩陣大小為30*3

% 07/09:增加正規化功能
% 注意:請時刻注意數據中的差值是否符合需求，請著重觀察數據的門檻值是否能區分想要的解答。


%最後修改日期 : 07/09


normalized_signal = normalize(signal, 'range');             %將數據正規化到 [0, 1] 範圍內
look_diff_signal=diff(normalized_signal);                   %找尋前後差值
positions = find(look_diff_signal > threshold);             %將差值大於門檻值之位置找出

for i=1:length(positions)
    %透過剛剛找出的位置將所需訊號提取出來
    output(:,i)=signal(positions(i)-find_num + 1:positions(i));
end