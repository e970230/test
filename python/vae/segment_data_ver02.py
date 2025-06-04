import numpy as np
import pandas as pd

def segment_data(daq_file, servo_file=None, nargout=2):
    """
    將 DAQ 與 servo 資料做切分處理。  
    若僅提供 daq_file（一個輸入），則只返回處理後的 DAQ 資料。  
    若同時提供 daq_file 與 servo_file，則：
       - 當 nargout==2 時，返回 (data_nidaq_Ms, servo_current_Ms)
       - 當 nargout==1 時，僅返回 data_nidaq_Ms
       
    參數說明:
      daq_file: Pandas DataFrame，第一欄為時間，其餘欄位為各通道資料。
      servo_file: Pandas DataFrame，第一欄為時間、第二欄為位置、第三欄為電流訊號（若有提供）。
      nargout: 決定返回輸出數目（僅在 servo_file 不為 None 時有效）。
    """
    #%% 讀取 DAQ 資料
    daqData = daq_file.iloc[:, 1:].values                # 取 ch.1~ch.7
    time_daqData = daq_file.iloc[:, 0].values.squeeze()    # 時間軸
    fs = 12800
    
    # 若只提供一個輸入，則僅返回 DAQ 資料處理結果（這裡可依需求修改）
    if servo_file is None:
        return daqData  # 或者 return time_daqData, daqData 等
    
    #%% 讀取 servo 資料（當第二個輸入有提供時）
    servo_time = servo_file.iloc[:, 0].values.squeeze()          # 第一欄：時間
    servo_pos = servo_file.iloc[:, 1].values.squeeze()           # 第二欄：位置
    servo_current = servo_file.iloc[:, 2].values.squeeze()       # 第三欄：電流訊號
    fs_servo = 4000

    #%% 位置訊號切分設定 (依預先規劃的區間)
    segments = np.array([
        [-20.0558, -128.0556],
        [-128.0556, -236.0556],
        [-452.0556, -560.0552]
    ])
    numM = segments.shape[0]  # 模型數目 (3)
    
    #%% 判斷去回程共有幾段 (以最後一個模型來判斷)
    inx01 = np.where((servo_pos <= segments[-1, 0]) & (servo_pos >= segments[-1, 1]))[0]
    inx02 = np.where(np.diff(inx01) > 1)[0]
    inx03 = inx01[inx02]
    N_total = len(inx03) + 1  # 去回程總段數

    #%% 依各模型切分索引
    inx_M_all = []  
    for h in range(numM):
        current_inx = np.where((servo_pos <= segments[h, 0]) & (servo_pos >= segments[h, 1]))[0]
        breaks = np.where(np.diff(current_inx) > 1)[0]
        if breaks.size > 0:
            end_points = current_inx[breaks]
        else:
            end_points = np.array([])

        seg_indices_list = []
        if current_inx.size == 0:
            seg_indices_list = []
        else:
            for k in range(N_total):
                if k == 0:
                    if end_points.size > 0:
                        seg = np.arange(current_inx[0], end_points[0] + 1)
                    else:
                        seg = current_inx
                    seg_indices_list.append(seg)
                elif k == N_total - 1:
                    if breaks.size > 0:
                        try:
                            start_val = current_inx[breaks[-1] + 1]
                            seg = np.arange(start_val, current_inx[-1] + 1)
                            seg_indices_list.append(seg)
                        except IndexError:
                            pass
                    else:
                        seg_indices_list.append(current_inx)
                else:
                    try:
                        start_val = current_inx[breaks[k - 1] + 1]
                        seg = np.arange(start_val, current_inx[breaks[k]] + 1)
                        seg_indices_list.append(seg)
                    except IndexError:
                        pass
        inx_M_all.append(seg_indices_list)
    
    # 取所有模型共有的最小段數 v
    num_segments = [len(seg_list) for seg_list in inx_M_all]
    v = min(num_segments)
    
    # 組成 inx4，維度 (模型數, 段數)，只保留前 v 段
    inx4 = [seg_list[:v] for seg_list in inx_M_all]
    
    # 僅保留去程段 (在 MATLAB 中為 1:2:(N-1)，這裡轉換成 0-index：0, 2, 4, …)
    take_inx = list(range(0, v - 1, 2))
    N_go = len(take_inx)
    
    #%% 切分電流與 nidaq 訊號 (依據計算出的索引)
    # 這裡用 2D list 儲存切分段資料，尺寸為 (去程段數, 模型數)
    servo_current_Ms = [[None for _ in range(numM)] for _ in range(N_go)]
    servo_time_Ms    = [[None for _ in range(numM)] for _ in range(N_go)]
    data_nidaq_Ms    = [[None for _ in range(numM)] for _ in range(N_go)]
    time_nidaq_Ms    = [[None for _ in range(numM)] for _ in range(N_go)]
    
    for h in range(numM):
        for i, seg_idx in enumerate(take_inx):
            segments_inx = inx4[h][seg_idx]
            # 取得該段在 servo_time 中的起點與終點時間
            seg_start_time = servo_time[segments_inx[0]]
            seg_end_time   = servo_time[segments_inx[-1]]
            segment_time = np.array([seg_start_time, seg_end_time])
            
            # 找出對應於邊界的索引
            servo_index_start = np.argmin(np.abs(servo_time - segment_time[0]))
            servo_index_end   = np.argmin(np.abs(servo_time - segment_time[1]))
            nidaq_index_start = np.argmin(np.abs(time_daqData - segment_time[0]))
            nidaq_index_end   = np.argmin(np.abs(time_daqData - segment_time[1]))
            
            inx_servo = np.arange(servo_index_start, servo_index_end + 1)
            inx_nidaq = np.arange(nidaq_index_start, nidaq_index_end + 1)
            
            servo_time_Ms[i][h]    = servo_time[inx_servo]
            servo_current_Ms[i][h] = servo_current[inx_servo]
            time_nidaq_Ms[i][h]    = time_daqData[inx_nidaq]
            data_nidaq_Ms[i][h]    = daqData[inx_nidaq, :]
    
    out1 = np.array(data_nidaq_Ms)
    out2 = np.array(servo_current_Ms)
    
    # 根據 nargout 決定返回一個或兩個輸出
    if nargout == 1:
        return out1
    else:
        return out1, out2

'''
# 測試用範例 (假設已有適當的 DataFrame 變數)
if __name__ == "__main__":
    # 假設從 CSV 檔讀取資料：
    daq_df = pd.read_csv('DAQ_data.csv', header=None)    # 根據檔案格式設定 header
    servo_df = pd.read_csv('servo_data.csv', header=None)
    
    # 呼叫時提供兩個輸入，並要求雙輸出
    data_nidaq, servo_current = segment_data(daq_df, servo_df, nargout=2)
    print("雙輸出結果 shape:")
    print("data_nidaq_Ms:", data_nidaq.shape)
    print("servo_current_Ms:", servo_current.shape)
    
    # 呼叫時只提供第一個輸入，僅返回 daq 資料處理結果
    only_data = segment_data(daq_df)
    print("單一輸出結果 shape:", only_data.shape)
'''    
