function [output]=diff_signal(signal,find_num,threshold)

% input--------------------------------------------------------------------
% signal:需要處理的訊號
% find_num:需要擷取訊號隻數量(若想取的訊號長度為50則此地方填50)
% threshold:尋找想排除的訊號之門檻值(當前後訊號差距超過此值時
%                                  將此變化之訊號位置註記下來
% output-------------------------------------------------------------------
% output:處理完的訊號(訊號*起伏數量)，亦即若今天訊號起伏處為三處且想取得這三處
%        前30個資料點之對應訊號，則此輸出矩陣大小為30*3

%最後修改日期 : 06/30

look_diff_signal=diff(signal);                   %找尋前後差值
positions = find(look_diff_signal > threshold);  %將差值大於門檻值之位置找出

for i=1:length(positions)
    %透過剛剛找出的位置將所需訊號提取出來
    output(:,i)=signal(positions(i)-find_num + 1:positions(i));
end