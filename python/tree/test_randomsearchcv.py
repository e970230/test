import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error
import scipy.io
import time
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('feature_dataset_top30.mat')

feature_dataset = mat['Data']
'''
# 读取文件
mat = scipy.io.loadmat('feature_dataset_heavy.mat')
feature_dataset = mat['feature_dataset']
'''
start_time = time.time()

init_data = feature_dataset[:, 1:]  # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]  # 擷取原數據的標籤，為原數據的第0列

init_data = init_data.astype(np.float64)
label = label.astype(np.float64)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2024/09/12註記若優化超參數使用到7個是極限，超過會太多

param_random = {
    'n_estimators': np.arange(500,2001,50),
    'learning_rate': np.linspace(0.01, 0.5, 5),
    'max_depth': np.arange(1,10),
    'min_child_weight': np.arange(1,10),
    'gamma': np.linspace(0.5,1.,10),
    'subsample': np.linspace(0.5,1.,10),
    'colsample_bytree': np.linspace(0.5,1.,10),
    'reg_alpha': np.linspace(0, 1, 10),
    'reg_lambda': np.linspace(0, 1, 10),
}


# 創建 XGBRegressor 模型
xgb_model = xgb.XGBRegressor(booster='gbtree', random_state=42)

#  創建 RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_model, param_random, 
                           n_iter=2000,scoring='neg_mean_squared_error', cv=skf, verbose=1, n_jobs=-1)

random_search.fit(init_data, label)



end_time = time.time()
elapsed_time = end_time - start_time

# 输出最佳参数和最佳得分
print("Elapsed Time: %.2f seconds" % elapsed_time)
print("Best Parameters:", random_search.best_params_)
print("Best Score:", -random_search.best_score_)  # 因为使用的是负的 MSE，所以需要取负值


best_model = random_search.best_estimator_

# 在验证数据上进行预测并计算 MSE
test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_mse = []
unique_numbers = np.unique(label)
final_vr_answer = np.empty((0, len(unique_numbers)))

for train_index, test_index in test_skf.split(init_data, label):
    X_train, X_test = init_data[train_index], init_data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
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

plt.figure(figsize=(10, 6))
plt.plot(random_search.cv_results_['mean_test_score'], color='blue', linestyle='--', marker='o', label='Mean Test Score')
plt.title('Grid Search Mean Test Score Over Parameter Combinations', fontsize=16)
plt.xlabel('Parameter Combination Index', fontsize=14)
plt.ylabel('Negative Mean Squared Error', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()