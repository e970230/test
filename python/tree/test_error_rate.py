import numpy as np
from sklearn.metrics import mean_squared_error

# 標準答案和預測答案
true_labels = np.array([120, 1100, 1500, 120, 1100, 1500, 120, 1100, 1500])
predicted_labels = np.array([110, 1005, 1450, 110, 1005, 1450, 100, 1005, 1450])

# 找出唯一的標籤
unique_labels = np.unique(true_labels)

# 計算每個標籤的誤差率
error_rates = {}

'''
for label in unique_labels:
    # 找出標準答案和預測答案中該標籤的索引
    true_indices = np.where(true_labels == label)[0]
    
    mse[label] = mean_squared_error(true_labels[true_indices], predicted_labels[true_indices])
'''
verify_mse = {label: mean_squared_error(true_labels[true_labels == label], predicted_labels[true_labels == label]) for label in unique_labels}

# 輸出每個標籤的誤差率
for label, verify_mse in verify_mse.items():
    print(f"標籤 {label} 的誤差率為 {verify_mse:.2f}")
