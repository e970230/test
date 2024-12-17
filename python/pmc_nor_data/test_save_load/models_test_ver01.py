import xgboost as xgb
import numpy as np
import pandas as pd
import json
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
csv_filename = 'best_solution_models_3.csv'  # 替換成你的 CSV 檔案名稱
df = pd.read_csv(csv_filename)

# 取得第 2 列的資料（以 0 為起始索引，2 對應索引 1）
second_row = df.iloc[9:-2,1].astype(int).values  # iloc[1] 取得第 2 列的資料

print("第 2 列的資料:")
print(second_row)


# ------------------------- 設定檔案名稱 -------------------------
model_name = 'models_3'  # 根據你的模型名稱設定
model_file = f'{model_name}_02.json'  # 模型檔案


new_test_data_file = 'Final_test_nor_input_data_models_3'   # 新測試數據檔案


# ------------------------- 載入模型與特徵設定 -------------------------
# 1. 載入 XGBoost 模型
loaded_model = xgb.XGBRegressor()
loaded_model.load_model(model_file)
print(f"模型已成功載入: {model_file}")
print("模型參數:", loaded_model.get_params())

# ------------------------- 處理新的測試數據 -------------------------
# 3. 讀取新測試數據
print("\n讀取新測試數據...")
new_test_data = scipy.io.loadmat(new_test_data_file)['Final_input_data']

# 過濾測試數據，僅保留選擇的特徵
filtered_test_data = new_test_data[:, second_row]
print(f"新測試數據維度（選取特徵後）: {filtered_test_data.shape}")

# ------------------------- 進行預測 -------------------------
print("\n進行預測...")
predictions = loaded_model.predict(filtered_test_data)

# ------------------------- 儲存預測結果 -------------------------
# 輸出預測結果
output_df = pd.DataFrame({'Predicted Values': predictions})
#output_df.to_csv(output_predictions_file, index=False)


# ------------------------- 顯示預測結果 -------------------------
print("\n預測結果:")
print(output_df.head())

# 繪製適應度趨勢圖
plt.figure(figsize=(10, 6))
plt.plot(predictions, color='blue', linestyle='--', label='Best Solution Fitness')
plt.title('GA Fitness Evolution Over Generations', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()