import numpy as np
import pandas as pd
import pickle
from feature_extarctor_ver02 import vibration_mix_current_features  #將抽特徵程式import進來
from segment_data_ver01 import segment_data

#----------------------前置說明-----------------------------
# 使用範例程式時須注意需準備以下檔案
# 1. 
# 2. 訓練好的模型檔案(.pkl檔)
# 3. 抽取特徵的編號及順序檔(.csv檔)
# 4. 


daq_file = pd.read_csv('DAQ_data.csv', header=None)
servo_file = pd.read_csv('servo_data.csv', header=None)

[data_nidaq_Ms,servo_current_Ms] = segment_data(daq_file, servo_file)


model_name = 1  # 根據你的模型名稱設定

# ------------------------- 載入模型與特徵設定 -------------------------

best_feature_models_1_F_10000 = pd.read_csv("best_solution_models_1_F_10000.csv")
best_feature_models_2_F_10000 = pd.read_csv("best_solution_models_2_F_10000.csv")
best_feature_models_3_F_10000 = pd.read_csv("best_solution_models_3_F_10000.csv")

# 1. 載入 XGBoost 模型
model_1 = pickle.load(open('models_1_F_10000.pkl', "rb"))
model_2 = pickle.load(open('models_2_F_10000.pkl', "rb"))
model_3 = pickle.load(open('models_3_F_10000.pkl', "rb"))



# 3. 讀取新測試數據
print("\n讀取新測試數據...")


#---------------------------振動+電流特徵抽取範例---------------------------
new_test_data = vibration_mix_current_features(
    spindle_fs = 10000 / (12 * 60),                                     #一倍頻
    spindle_fs_target = np.arange(4, 32 + 1, 4),                        #振動訊號尋找倍頻的目標 ex.4:4:32 4~32倍頻
    current_data_spindle_fs_target = np.arange(4, 32 + 0.1, 0.5),       #電流訊號尋找倍頻的目標 ex.# 4:0.5:32 4~32倍頻
    BD_delta_F = 20,                                                    #頻帶能量間隔，單位HZ
    BD_spindle_start = 18,                                              #振動訊號頻帶能量起始倍數 ex.若想從18倍頻開始找則輸入 18
    BD_spindle_end = 200,                                               #振動訊號頻帶能量結束倍數 ex.若想在200倍頻結束找則輸入 200
    current_spindle_start = 18,                                         #電流訊號頻帶能量起始倍數 ex.若想從18倍頻開始找則輸入 18
    current_spindle_end = 50,                                           #電流訊號頻帶能量結束倍數 ex.若想在50倍頻結束找則輸入 50
    tolerance = 5,                                                      #尋找特定頻率之前後誤差的閥值，5代表5HZ
    fs = 12800,                                                         #振動訊號採樣頻率
    fs_servo = 4000,                                                    #電流訊號採樣頻率
    ALL_data_nidaq = data_nidaq_Ms[:, model_name-1],                           #原始振動訊號數據(實驗趟數*models(此範例models的意思為指定哪一個models))(實驗趟數*all_ch)
    ALL_servo_current = servo_current_Ms[:, model_name-1]                      #原始電流訊號數據(實驗趟數*models(此範例models的意思為指定哪一個models))(實驗趟數*all_ch)
)


if model_name == 1:
    second_row = best_feature_models_1_F_10000.iloc[9:-2,1].astype(int).values  # iloc[1] 取得第 2 列的資料

if model_name == 2:
    second_row = best_feature_models_2_F_10000.iloc[9:-2,1].astype(int).values  # iloc[1] 取得第 2 列的資料

if model_name == 3:
    second_row = best_feature_models_3_F_10000.iloc[9:-2,1].astype(int).values  # iloc[1] 取得第 2 列的資料

# 過濾測試數據，僅保留選擇的特徵
filtered_test_data = new_test_data[:, second_row]
print(f"新測試數據維度（選取特徵後）: {filtered_test_data.shape}")


# ------------------------- 進行預測 -------------------------
print("\n進行預測...")
if model_name == 1:
    predictions = model_1.predict(filtered_test_data)

if model_name == 2:
    predictions = model_2.predict(filtered_test_data)

if model_name == 3:
    predictions = model_3.predict(filtered_test_data)

print('最終預測平均值單位(um)')
print(np.mean(predictions))
