import numpy as np
import pandas as pd
import pickle
from feature_extarctor_ver02 import vibration_mix_current_features  # 抽特徵程式
from segment_data_ver01 import segment_data # 分割訊號程式

# ---------------------- 前置說明 -----------------------------
# 使用本範例程式時，請準備下列檔案：
# 1. DAQ_data.csv 與 servo_data.csv（原始數據）
# 2. 訓練好的模型檔案 (.pkl)
# 3. 最佳特徵設定檔案 (.csv)

# ---------------------- 讀取原始數據 -----------------------------
daq_file = pd.read_csv('DAQ_data.csv', header=None)
servo_file = pd.read_csv('servo_data.csv', header=None)

[data_nidaq_Ms, servo_current_Ms] = segment_data(daq_file, servo_file)

# ---------------------- 設定要進行預測的模型 ----------------------
model_list = [1, 2, 3]
# 若只想預測模型1跟3，則設定：
# model_list = [1, 3]  
# 若只想預測模型2，則設定：
# model_list = [2]

# ---------------------- 載入最佳特徵設定 ----------------------
best_feature_models_1_F_10000 = pd.read_csv("best_solution_models_1_F_10000.csv")
best_feature_models_2_F_10000 = pd.read_csv("best_solution_models_2_F_10000.csv")
best_feature_models_3_F_10000 = pd.read_csv("best_solution_models_3_F_10000.csv")

# ---------------------- 載入模型 ----------------------
model_1 = pickle.load(open('models_1_F_10000.pkl', "rb"))
model_2 = pickle.load(open('models_2_F_10000.pkl', "rb"))
model_3 = pickle.load(open('models_3_F_10000.pkl', "rb"))

# 建立字典以方便後續存取
models_dict = {1: model_1, 2: model_2, 3: model_3}
best_features_dict = {
    1: best_feature_models_1_F_10000,
    2: best_feature_models_2_F_10000,
    3: best_feature_models_3_F_10000
}

# 用於儲存各模型預測結果的字典
predictions_dict = {}

# ---------------------- 逐一對指定模型進行預測 ----------------------
for model_num in model_list:
    print(f"\n開始處理模型 {model_num} 的預測...")

    # ---------------------- 特徵抽取 ----------------------
    # 注意：此處將傳入對應模型通道的數據
    new_test_data = vibration_mix_current_features(
        spindle_fs = 10000 / (12 * 60),                              # 一倍頻
        spindle_fs_target = np.arange(4, 32 + 1, 4),                 # 振動訊號尋找倍頻目標
        current_data_spindle_fs_target = np.arange(4, 32 + 0.1, 0.5),  # 電流訊號尋找倍頻目標
        BD_delta_F = 20,                                             # 頻帶能量間隔(Hz)
        BD_spindle_start = 18,                                       # 振動訊號頻帶能量起始倍數
        BD_spindle_end = 200,                                        # 振動訊號頻帶能量結束倍數
        current_spindle_start = 18,                                  # 電流訊號頻帶能量起始倍數
        current_spindle_end = 50,                                    # 電流訊號頻帶能量結束倍數
        tolerance = 5,                                               # 頻率尋找容許誤差(Hz)
        fs = 12800,                                                  # 振動訊號採樣頻率
        fs_servo = 4000,                                             # 電流訊號採樣頻率
        ALL_data_nidaq = data_nidaq_Ms[:, model_num-1],               # 選取對應模型的振動數據(注意：Python從0開始索引)
        ALL_servo_current = servo_current_Ms[:, model_num-1]          # 選取對應模型的電流數據
    )

    # ---------------------- 根據模型選取最佳特徵 ----------------------
    # 此處假設最佳特徵 CSV 檔的資料結構與原本一致
    best_feature_df = best_features_dict[model_num]
    # 取出第 10 列至倒數第3列的特徵索引 
    feature_indices = best_feature_df.iloc[9:-2, 1].astype(int).values

    # 篩選測試數據，只保留選取的特徵
    filtered_test_data = new_test_data[:, feature_indices]
    print(f"模型 {model_num} 預測用數據維度 (選取特徵後): {filtered_test_data.shape}")

    # ---------------------- 進行預測 ----------------------
    current_model = models_dict[model_num]
    predictions = current_model.predict(filtered_test_data)
    mean_prediction = np.mean(predictions)
    print(f"模型 {model_num} 預測結果平均值 (um): {mean_prediction}")

    # 儲存預測結果
    predictions_dict[model_num] = predictions

