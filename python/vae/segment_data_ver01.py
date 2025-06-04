import numpy as np
import pandas as pd

def segment_data(daq_file,servo_file):
    # 讀取資料
    # 載入 DAQ_data.mat (包含 daqData, time_daqData, fs)
    
    daqData = daq_file.iloc[:,1:].values                # ch.1~ch.7
    time_daqData = daq_file.iloc[:,0].values.squeeze()  # 時間
    fs = 12800
    
    # 載入 servo_data.mat (包含 servo_time, servo_current, servo_pos, fs_servo)
    
    servo_time = servo_file.iloc[:,0].values.squeeze()          #時間為csv檔第一列
    servo_current = servo_file.iloc[:,2].values.squeeze()       #電流訊號為csv檔第三列
    servo_pos = servo_file.iloc[:,1].values.squeeze()           #位置訊號為csv檔第二列
    fs_servo = 4000
    
    #%% 位置訊號切分的設定
    # 根據預先規劃的切分位置，定義各區間 (每一列: [起點, 終點])
    segments = np.array([
        [-20.0558, -128.0556],
        [-128.0556, -236.0556],
        [-452.0556, -560.0552]
    ])
    numM = segments.shape[0]  # 模型數目 (3)
    
    #判斷去回程共有幾段
    # 以最後一個模型 (model 3) 為例，找出 servo_pos 在該區間內的索引
    inx01 = np.where((servo_pos <= segments[-1, 0]) & (servo_pos >= segments[-1, 1]))[0]
    # 找出不連續處（相鄰索引差值大於1）的部分
    inx02 = np.where(np.diff(inx01) > 1)[0]
    inx03 = inx01[inx02]
    N_total = len(inx03) + 1  # 去回程總段數

    # 針對每個模型依據位置範圍進行切分
    # 利用 inx_M_all 儲存每個模型內所有連續段的索引（以 servo_pos 的索引計）
    inx_M_all = []  
    for h in range(numM):
        # 取得在指定位置範圍內的所有索引
        current_inx = np.where((servo_pos <= segments[h, 0]) & (servo_pos >= segments[h, 1]))[0]
        # 找出連續性斷點
        breaks = np.where(np.diff(current_inx) > 1)[0]
        if breaks.size > 0:
            end_points = current_inx[breaks]
        else:
            end_points = np.array([])

        seg_indices_list = []
        if current_inx.size == 0:
            # 若沒有資料則加入空列表
            seg_indices_list = []
        else:
            # 將 current_inx 根據連續性切成 N_total 段
            for k in range(N_total):
                if k == 0:
                    # 第一段：從第一個索引到第一個不連續處
                    if end_points.size > 0:
                        seg = np.arange(current_inx[0], end_points[0] + 1)
                    else:
                        seg = current_inx  # 若只有一段
                    seg_indices_list.append(seg)
                elif k == N_total - 1:
                    # 最後一段：從最後一個斷點後到最後一筆資料
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
                    # 中間段：從上一個斷點後到下一個斷點處
                    try:
                        start_val = current_inx[breaks[k - 1] + 1]
                        seg = np.arange(start_val, current_inx[breaks[k]] + 1)
                        seg_indices_list.append(seg)
                    except IndexError:
                        pass
        inx_M_all.append(seg_indices_list)
    
    # 檢查每個模型的段數，若不同則取最小的段數 (v)
    num_segments = [len(seg_list) for seg_list in inx_M_all]
    v = min(num_segments)
    
    # 建立 inx4：維度為 (模型數, 段數)，只保留每個模型前 v 段
    inx4 = [seg_list[:v] for seg_list in inx_M_all]
    
    # 僅保留去程段 (去程為奇數段)
    # take_inx = 1:2:(N-1)；轉換成 Python (0-indexed) 為：0, 2, 4, ... (但不含最後一段)
    take_inx = list(range(0, v - 1, 2))
    N_go = len(take_inx)
    
    # 切分電流與 nidaq 訊號 (依照切分的索引)
    # 建立 2D list 儲存各切分段資料，維度： (去程段數, 模型數)
    servo_current_Ms = np.array([[None for _ in range(numM)] for _ in range(N_go)])
    servo_time_Ms    = np.array([[None for _ in range(numM)] for _ in range(N_go)])
    data_nidaq_Ms    = np.array([[None for _ in range(numM)] for _ in range(N_go)])
    time_nidaq_Ms    = np.array([[None for _ in range(numM)] for _ in range(N_go)])
    
    for h in range(numM):
        for i, seg_idx in enumerate(take_inx):
            # 取得 model h 中第 seg_idx 段的索引 (對應 servo_time 與 servo_current)
            segments_inx = inx4[h][seg_idx]
            # 取得該段在 servo_time 中的起點與終點時間
            seg_start_time = servo_time[segments_inx[0]]
            seg_end_time   = servo_time[segments_inx[-1]]
            segment_time = np.array([seg_start_time, seg_end_time])
            
            # 找出 servo_time 與 time_daqData 中與該段邊界最接近的索引
            servo_index_start = np.argmin(np.abs(servo_time - segment_time[0]))
            servo_index_end   = np.argmin(np.abs(servo_time - segment_time[1]))
            nidaq_index_start = np.argmin(np.abs(time_daqData - segment_time[0]))
            nidaq_index_end   = np.argmin(np.abs(time_daqData - segment_time[1]))
            
            # 產生包含起始與終點 (inclusive) 的 index range
            inx_servo = np.arange(servo_index_start, servo_index_end + 1)
            inx_nidaq = np.arange(nidaq_index_start, nidaq_index_end + 1)
            
            # 將切分好的資料儲存
            servo_time_Ms[i][h]    = servo_time[inx_servo]
            servo_current_Ms[i][h] = servo_current[inx_servo]
            time_nidaq_Ms[i][h]    = time_daqData[inx_nidaq]
            data_nidaq_Ms[i][h]    = daqData[inx_nidaq, :]
    

    return np.array(data_nidaq_Ms) , np.array(servo_current_Ms)

