from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import time

mat = scipy.io.loadmat('feature_dataset_heavy.mat')


print(mat.keys())


feature_dataset = mat['feature_dataset']


data = feature_dataset[:, 1:]  # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]  # 擷取原數據的標籤，為原數據的第0列


#將數據百分之20做為測試集百分之80做為訓練集
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20)


M = 100  # 樹的數量
nmin = 3  # 葉節點最小樣本數

# 開始計時
start_time = time.time()

# 初始化 ExtraTreesRegressor 模型
etr = ExtraTreesRegressor(n_estimators=M, min_samples_leaf=nmin, random_state=42, verbose=1) #verbose=1:把建立的樹時的資訊給顯示出來
#etr = ExtraTreesRegressor(n_estimators=M, bootstrap=True, max_samples=k, min_samples_leaf=nmin, random_state=42, verbose=1)
print(etr)

# 利用訓練資料來擬和模型
etr.fit(X_train, y_train)
score = etr.score(X_train, y_train)
print("Training Score: ", score)

# 預測測試數據
y_pred = etr.predict(X_test)

# 計算預測答案和原始標籤之MSE和RMSE值
mse = mean_squared_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % mse**(0.5))

# 計算和顯示運行時間
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed Time: %.2f seconds" % elapsed_time)

# Plot the results
x_ax = range(len(y_test))
plt.figure()
plt.plot(x_ax, y_test, lw=0.6, color="blue", label="Original")
plt.plot(x_ax, y_pred, lw=0.8, color="red", label="Predicted")
plt.title("ExtraTreesRegressor: Original vs Predicted")
plt.xlabel("Data Points")
plt.ylabel("Target Variable")
plt.legend()
plt.show()
