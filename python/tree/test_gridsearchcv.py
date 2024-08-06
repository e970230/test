import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


init_data = feature_dataset[:, 1:]  # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]  # 擷取原數據的標籤，為原數據的第0列

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

param_grid = {
    'n_estimators': np.arange(10,401,10),
    'learning_rate': np.arange(0.01,0.3,0.01),
    'max_depth': np.arange(1,10),
    'min_child_weight': np.arange(1,10),
    'gamma': np.arange(0,0.5,0.1),
    'subsample': np.arange(0.5,1,0.1),
    'colsample_bytree': np.arange(0.5,1,0.1),
    'colsample_bylevel': np.arange(0.5,1,0.1),
    'colsample_bynode': np.arange(0.5,1,0.1),
    'lambda': np.arange(0,5,0.1),
    'alpha': np.arange(0,5,0.1)
}

# 创建 XGBRegressor 模型
xgb_model = xgb.XGBRegressor(booster='gbtree', random_state=42,learning_rate=0.2)

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=skf, verbose=1, n_jobs=-1)

# 记录开始时间
start_time = time.time()

# 执行网格搜索
grid_search.fit(init_data, label)

# 记录结束时间
end_time = time.time()
elapsed_time = end_time - start_time

# 输出最佳参数和最佳得分
print("Elapsed Time: %.2f seconds" % elapsed_time)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", -grid_search.best_score_)  # 因为使用的是负的 MSE，所以需要取负值

# 使用最佳参数重新训练模型并验证
best_model = grid_search.best_estimator_

# 在验证数据上进行预测并计算 MSE
test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
all_mse = []

for train_index, test_index in test_skf.split(init_data, label):
    X_train, X_test = init_data[train_index], init_data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    all_mse.append(mse)

final_mse_mean = np.mean(all_mse, axis=0)
unique_numbers = np.unique(label)
verify_mse = {er_label: mean_squared_error(y_test[y_test == er_label], y_pred[y_test == er_label]) for er_label in unique_numbers}


# 輸出每個標籤的mse值
for er_label, verify_mse in verify_mse.items():
    if verify_mse>=er_label:
        er_answer = " 誤差率過大 "
    else:
        er_answer = f" 誤差率百分之{verify_mse/er_label*100} "
    print(f"預壓力 {er_label} 的MSE值為 {verify_mse:.2f} {er_answer}")


print("預測模型驗證 MSE 值:", final_mse_mean)


plt.figure(figsize=(10, 6))
plt.plot(grid_search.cv_results_['mean_test_score'], color='blue', linestyle='--', marker='o', label='Mean Test Score')
plt.title('Grid Search Mean Test Score Over Parameter Combinations', fontsize=16)
plt.xlabel('Parameter Combination Index', fontsize=14)
plt.ylabel('Negative Mean Squared Error', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()
