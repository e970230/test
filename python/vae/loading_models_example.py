import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import scipy.io
import pandas as pd
import scipy.io as sio
from feature_extarctor_ver02 import vibration_mix_current_features  #將抽特徵程式import進來
from segment_data_ver01 import segment_data

#----------------------前置說明-----------------------------
# 使用範例程式時須注意需準備以下檔案
# 1. 準備預測的數據，可以是此範例已經處理好的"test_input_models_1_F_10000.mat"，或者自行使用特徵抽取程式後的數據帶入
# 2. 訓練好的模型檔案(.pkl檔)
# 3. 抽取特徵的編號及順序檔(.csv檔)
# 4. 原位移計的中位數訊號，此範例有以處理完的"test_output_models_1_F_10000.mat"為範例


daq_file = pd.read_csv('DAQ_data.csv', header=None)
servo_file = pd.read_csv('servo_data.csv', header=None)

[data_nidaq_Ms,servo_current_Ms] = segment_data(daq_file,servo_file)


model_name = 3  # 根據你的模型名稱設定

# ------------------------- 載入模型與特徵設定 -------------------------

best_feature_models_1_F_10000 = pd.read_csv("best_solution_models_1_F_10000.csv")
best_feature_models_2_F_10000 = pd.read_csv("best_solution_models_2_F_10000.csv")
best_feature_models_3_F_10000 = pd.read_csv("best_solution_models_3_F_10000.csv")

# 1. 載入 XGBoost 模型
model_1 = pickle.load(open('models_1_F_10000.pkl', "rb"))
model_2 = pickle.load(open('models_2_F_10000.pkl', "rb"))
model_3 = pickle.load(open('models_3_F_10000.pkl', "rb"))


# ------------------------- 處理新的測試數據 -------------------------
# 3. 讀取新測試數據
print("\n讀取新測試數據...")
#mat_data = sio.loadmat("C:/Users/user/Desktop/desktop_pmc/訊號資料/dataset_Test1_F10000.mat")
#ALL_data_nidaq = mat_data.get("ALL_data_nidaq", None)
#ALL_servo_current = mat_data.get("ALL_servo_current", None)

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

#file_path = r"C:/Users/user/Desktop/desktop_pmc/訊號資料/資料_2_70_30_整理完版本/訓練資料/test_input_models_1_F_10000.mat"
#new_test_data = scipy.io.loadmat(file_path)['test_input_data']

# 取得第 2 列的資料（以 0 為起始索引，2 對應索引 1）
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



# === 2. 每組資料「不跨組」地以5個樣本為一段去計算平均 ===

window_size = 5

all_means = []    # 紀錄所有平均值
all_x = []        # 紀錄對應的 x 座標
current_x = 0     # 用來累積 x 座標的索引


# 在該組資料內，每次只往右移 1
for start_idx in range(len(predictions) - window_size + 1):
    end_idx = start_idx + window_size  # end_idx - start_idx = 5
    chunk = predictions[start_idx:end_idx]
    chunk_mean = np.mean(chunk)
    
    all_means.append(chunk_mean)
    all_x.append(current_x)
    current_x += 1  # x座標往右移 1

# === 3. 畫出「原始 6 組」與「每 5 個樣本平均後」的對照圖 ===
plt.figure(figsize=(10, 6))



# 再畫出「每 5 個樣本平均」的結果
plt.plot(range(len(all_means)), all_means, 'rx', label='pred_data (chunk=5)', markersize=5)
#plt.plot(range(len(answer)), answer,'bo',markerfacecolor='none', label='answer (chunk=5)',markersize=2)

plt.title('Mean of Every 5 Samples in sample data')
plt.xlabel('sample')
plt.ylabel('MSE')
plt.legend()
plt.tight_layout()
plt.savefig(f'{model_name}_mean_predicted.jpg')  # 或 .jpg、.pdf、.svg 等
plt.show()
