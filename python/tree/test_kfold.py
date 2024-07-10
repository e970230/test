import pygad
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
import scipy.io
from sklearn.metrics import mean_squared_error

mat = scipy.io.loadmat('feature_dataset_heavy.mat')

feature_dataset = mat['feature_dataset']


data = feature_dataset[:, 1:]  # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]  # 擷取原數據的標籤，為原數據的第0列

def Customized_kfold(data,label):
    unique_numbers = np.unique(label)
    for fold in unique_numbers:
    
        # 切分數據
        X_train, X_test = data[np.where(label != fold)[0]], data[np.where(label == fold)[0]]
        y_train, y_test = label[np.where(label != fold)[0]], label[np.where(label == fold)[0]]
    return X_train,X_test,y_train,y_test


unique_numbers = np.unique(label)


all_mse = []
# 自行實現KFold 交叉驗證
for fold in unique_numbers:
    
    X_train, X_test = data[np.where(label != fold)[0]], data[np.where(label == fold)[0]]
    y_train, y_test = label[np.where(label != fold)[0]], label[np.where(label == fold)[0]]
    # 創建 ExtraTreesRegressor 模型
    model = ExtraTreesRegressor(n_estimators=200,          #將第一個解作為樹的數量
                                max_features=25,          #將第二個解作為分裂時考慮的最大特徵數
                                min_samples_split=10,     #將第三個解作為葉節點最小樣本數
                                random_state=42)
    # 使用模型進行預測
    model.fit(X_train, y_train)                 #訓練模型
    y_pred = model.predict(X_test)              #預測解答

    # 計算預測答案和原始標籤之MSE值
    mse = mean_squared_error(y_test, y_pred)    #計算MSE值
    all_mse.append(mse)                         #將當次MSE值記錄下來
    
final_mse_mean = np.mean(all_mse, axis=0)
    

print(f"\n{np.shape(unique_numbers)} 折交叉驗證的平均準確度: {final_mse_mean}")







'''

#kf = KFold(n_splits=5, shuffle=False, random_state=None)
skf = StratifiedKFold(n_splits=3, shuffle=False, random_state=None)

M = 100  # 樹的數量
nmin = 3  # 葉節點最小樣本數

all_mse = []

fold = 1
for train_index, test_index in skf.split(data,label):
    print(f"Fold {fold} - Training indices: {train_index}")
    print(f"Fold {fold} - Testing indices: {test_index}")
    
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    etr = ExtraTreesRegressor(n_estimators=M, min_samples_leaf=nmin, random_state=42, verbose=1) #verbose=1:把建立的樹時的資訊給顯示出來
    etr.fit(X_train, y_train)
    y_pred = etr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    all_mse.append(mse)
    print("Fold %d - MSE: %.2f" % (fold, mse))
    fold += 1


print(all_mse)

final_mse_mean = np.mean(all_mse, axis=0)

print(final_mse_mean)

'''

