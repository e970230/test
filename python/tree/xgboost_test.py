import pygad
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import scipy.io
import time
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('feature_dataset_top30.mat')

feature_dataset = mat['Data']

data = feature_dataset[:, 1:]      # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]           # 擷取原數據的標籤，為原數據的第0列

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)     #設定sKfold交叉驗證模組(拆分的組數，再拆分前是否打亂，隨機性設定)

all_mse = []



unique_numbers = np.unique(label)       #將標籤中不一樣處給區別出來，以後續處理使用

final_vr_answer = np.empty((0, len(unique_numbers)))

model = xgb.XGBRegressor(n_estimators= 47,             #將第一個解作為樹的數量
                             learning_rate= 0.13,     #將第三個解作為學習率
                             max_depth=8,                 #將第二個解作為樹的最大深度
                             min_child_weight=4,
                             gamma = 0.27,
                             subsample = 0.99,
                             colsample_bytree = 0.69,
                             reg_alpha = 0.53,
                             booster='gbtree',
                             random_state=42)
for train_index, test_index in skf.split(data,label):
    X_train, X_test = data[train_index], data[test_index]               #將數據拆分成訓練數據和測試數據，並透過Kfold交叉驗證方式進行區分
    y_train, y_test = label[train_index], label[test_index]             #將標籤拆分成訓練標籤和測試標籤，並透過Kfold交叉驗證方式進行區分
    
    # 使用模型進行預測
    model.fit(X_train, y_train)                 #訓練模型
    y_pred = model.predict(X_test)              #預測解答

    # 計算預測答案和原始標籤之MSE值
    #mse = mean_squared_error(y_test, y_pred)    #計算MSE值
    #all_mse.append(mse)                         #將當次MSE值記錄下來
    verify_mse_test = {er_label: mean_squared_error(y_test[y_test == er_label], y_pred[y_test == er_label]) for er_label in unique_numbers}
    verify_mse = verify_mse_test
    temporary_vr_answer = []  # 初始化為一個空列表，存儲每次的vr_answer

    for er_label, verify_mse in verify_mse.items():
        if verify_mse >= er_label:
            vr_answer = 1000
        else:
            vr_answer = verify_mse / er_label * 100

        temporary_vr_answer.append(vr_answer)  # 使用列表的append方法將vr_answer追加到final_vr_answer中
    

    temporary_vr_answer = np.array(temporary_vr_answer).reshape(1, -1)  # 將其轉換為一行
    final_vr_answer = np.vstack((final_vr_answer, temporary_vr_answer)) # 垂直堆疊

    mse = mean_squared_error(y_test, y_pred)
    all_mse.append(mse)

#final_mse_mean = np.mean(all_mse, axis=0)       #將所有記錄下來的MSE值進行平均

row_means = np.mean(final_vr_answer, axis=0)
final_mse_mean = np.mean(all_mse, axis=0)

# 輸出每個標籤的mse值
# 假設 unique_numbers 是你的標籤對應的數據
for er_label, row_mean in zip(unique_numbers, row_means):
    if row_mean >= er_label:
        er_answer = "誤差率過大"
    else:
        er_answer = f"誤差率百分之 {row_mean / er_label * 100:.2f}"

    print(f"預壓力 {er_label} 的MSE值為 {row_mean:.2f} {er_answer}")

print("預測模型驗證MSE值:", final_mse_mean)
