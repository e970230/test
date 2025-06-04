import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import time

# 產生合成迴歸數據：200,000 筆資料、200 個特徵
X, y = make_regression(n_samples=200000, n_features=200, noise=0.1, random_state=42)

n_estimators = 200

# ---------------------- CPU 模式 ----------------------
print("使用 CPU 訓練 XGBoost 迴歸樹 ...")
model_cpu = xgb.XGBRegressor(
    n_estimators=n_estimators,
    tree_method='hist',           # CPU 使用 hist 演算法
    predictor='cpu_predictor',    # CPU 預測器
    random_state=42
)

start_cpu = time.time()
model_cpu.fit(X, y)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

y_pred_cpu = model_cpu.predict(X)
mse_cpu = mean_squared_error(y, y_pred_cpu)

print("CPU 訓練時間: {:.4f} 秒".format(cpu_time))
print("CPU 預測 MSE: {:.4f}".format(mse_cpu))


# ---------------------- GPU 模式 ----------------------
print("\n使用 GPU 訓練 XGBoost 迴歸樹 ...")
# 注意：請確認你的系統已正確安裝 CUDA 及 GPU 版本的 xgboost
model_gpu = xgb.XGBRegressor(
    n_estimators=n_estimators,
    device='cuda',                # 保留最新的 GPU 設定
    random_state=42
)

start_gpu = time.time()
model_gpu.fit(X, y)
# 若 GPU 運算為非同步，可考慮同步（例如用 cupy 或其他方法）確保計時正確
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

y_pred_gpu = model_gpu.predict(X)
mse_gpu = mean_squared_error(y, y_pred_gpu)

print("GPU 訓練時間: {:.4f} 秒".format(gpu_time))
print("GPU 預測 MSE: {:.4f}".format(mse_gpu))
