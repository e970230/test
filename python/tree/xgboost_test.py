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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)     #設定sKfold交叉驗證模組(拆分的組數，再拆分前是否打亂，隨機性設定)

all_mse = []

unique_numbers = np.unique(label)       #將標籤中不一樣處給區別出來，以後續處理使用

model = xgb.XGBRegressor(n_estimators=300,          #將第一個解作為樹的數量
                             max_depth=20,             #將第二個解作為樹的最大深度
                             learning_rate=0.15,              #將第三個解作為學習率
                             booster='gbtree',
                             min_child_weight=10,
                             random_state=42)
for train_index, test_index in skf.split(data,label):
    X_train, X_test = data[train_index], data[test_index]               #將數據拆分成訓練數據和測試數據，並透過Kfold交叉驗證方式進行區分
    y_train, y_test = label[train_index], label[test_index]             #將標籤拆分成訓練標籤和測試標籤，並透過Kfold交叉驗證方式進行區分
    
    # 使用模型進行預測
    model.fit(X_train, y_train)                 #訓練模型
    y_pred = model.predict(X_test)              #預測解答

    # 計算預測答案和原始標籤之MSE值
    mse = mean_squared_error(y_test, y_pred)    #計算MSE值
    all_mse.append(mse)                         #將當次MSE值記錄下來

final_mse_mean = np.mean(all_mse, axis=0)       #將所有記錄下來的MSE值進行平均

verify_mse = {er_label: mean_squared_error(y_test[y_test == er_label], y_pred[y_test == er_label]) for er_label in unique_numbers}

# 輸出每個標籤的mse值
for er_label, verify_mse in verify_mse.items():
    print(f"預壓力 {er_label} 的MSE值為 {verify_mse:.2f}")

print("預測模型驗證MSE值:",final_mse_mean)