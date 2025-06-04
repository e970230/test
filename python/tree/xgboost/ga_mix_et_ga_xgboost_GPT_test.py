import pygad
import numpy as np
import cupy as cp
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import scipy.io
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

#---------------------------------------------------------------
# 這份程式透過 pygad 的 GA 優化 XGBRegressor 超參數，
# 並將所有數據在讀入後預先轉移到 GPU，運算（訓練、預測及評估）全在 GPU 上執行，
# 避免 CPU 與 GPU 間的頻繁轉換。
#---------------------------------------------------------------

def fitness_func_method_two(ga_instance, solution, solution_idx):
    # 根據是否選擇特徵決定使用全部或部分數據：
    if num_params == 0:
        data_gpu = init_data_gpu
    else:
        selected_features = solution[9:]
        data_gpu = init_data_gpu[:, selected_features]
    
    # 建立 XGBRegressor 模型 (保留 device="cuda" 設定)
    model = xgb.XGBRegressor(n_estimators=solution[0],
                             learning_rate=solution[1] * 0.01,
                             max_depth=solution[2],
                             min_child_weight=solution[3],
                             gamma=solution[4] * 0.01,
                             subsample=solution[5] * 0.01,
                             colsample_bytree=solution[6] * 0.01,
                             reg_lambda=solution[7] * 0.01,
                             reg_alpha=solution[8] * 0.01,
                             booster='gbtree',
                             device="cuda",
                             random_state=42)
    
    def train_and_evaluate(train_idx, test_idx):
        # 直接從預先在 GPU 上的數據中切片
        X_train_gpu = data_gpu[train_idx]
        X_test_gpu = data_gpu[test_idx]
        y_train_gpu = label_gpu[train_idx]
        y_test_gpu = label_gpu[test_idx]
        
        model.fit(X_train_gpu, y_train_gpu)
        y_pred = model.predict(X_test_gpu)
        # 將預測結果轉成 cupy 陣列
        y_pred = cp.asarray(y_pred)
        # 全部在 GPU 上計算 MSE，最後轉為 float 返回
        mse = cp.mean((y_test_gpu - y_pred) ** 2)
        return mse.item()
    
    # 使用單執行緒遍歷交叉驗證分割
    all_mse = []
    for train_idx, test_idx in skf.split(data_gpu, label):
        mse = train_and_evaluate(train_idx, test_idx)
        all_mse.append(mse)
    
    return -np.mean(all_mse)

def generate_dynamic_gene_space(num_params, init_data):
    Dimensions = np.shape(init_data)
    predefined_ranges = [{'low': 1, 'high': Dimensions[1]} for _ in range(num_params)]
    return predefined_ranges

def generate_all_or_number(num_params, gene_space, init_data):
    gene_space_next = generate_dynamic_gene_space(num_params, init_data)
    final_gene_space = gene_space + gene_space_next if gene_space_next else gene_space
    return final_gene_space

def on_generation(ga_instance):
    print(f"第 {ga_instance.generations_completed} 代")
    print(f"最佳適應度值： {ga_instance.best_solutions_fitness[ga_instance.generations_completed-1]}")
    print("")

#---------------------------------------------------------------
# 主要程式區域
#---------------------------------------------------------------

mat = scipy.io.loadmat('feature_dataset_heavy.mat')
feature_dataset = mat['feature_dataset']  # 格式：樣本 × 特徵

init_data = feature_dataset[:, 1:]  # CPU 版特徵（用於分割、索引）
label = feature_dataset[:, 0]         # CPU 版標籤

# 預先將所有數據轉到 GPU 上，這裡只做一次
init_data_gpu = cp.asarray(init_data)
label_gpu = cp.asarray(label)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
unique_numbers = np.unique(label)

# GA 參數設定
num_generations = 10       # 疊代次數
num_parents_mating = 4     # 每代交配染色體數量
sol_per_pop = 5            # 染色體總數
num_params = 50            # 選取的特徵數量 (0 表示全部特徵)
num_genes = 9 + num_params # 總參數數

gene_space = [
    {'low': 500, 'high': 1500},  # n_estimators
    {'low': 1, 'high': 50},      # learning_rate
    {'low': 1, 'high': 20},      # max_depth
    {'low': 1, 'high': 10},      # min_child_weight
    {'low': 0, 'high': 50},      # gamma
    {'low': 50, 'high': 100},    # subsample
    {'low': 50, 'high': 100},    # colsample_bytree
    {'low': 0, 'high': 100},     # reg_lambda
    {'low': 0, 'high': 100},     # reg_alpha
]

final_gene_space = generate_all_or_number(num_params, gene_space, init_data)

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func_method_two,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=final_gene_space,
    gene_type=int,
    parent_selection_type="rank",
    crossover_type="single_point",
    mutation_type="random",
    mutation_probability=0.3,
    on_generation=on_generation,
    random_seed=42,
    save_solutions=False
)

start_time = time.time()
ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed Time: {:.2f} seconds".format(elapsed_time))
print("最佳超參數組合:", solution)
print("最佳解的適應度值:", solution_fitness)

#---------------------------------------------------------------
# 測試階段（全 GPU 運算）
#---------------------------------------------------------------
if num_params == 0:
    test_data_cpu = init_data
    test_data_gpu = init_data_gpu
else:
    selected_features = solution[9:]
    test_data_cpu = init_data[:, selected_features]
    test_data_gpu = init_data_gpu[:, selected_features]

test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_mse = []

for train_idx, test_idx in test_skf.split(test_data_cpu, label):
    X_train_gpu = test_data_gpu[train_idx]
    X_test_gpu  = test_data_gpu[test_idx]
    y_train_gpu = label_gpu[train_idx]
    y_test_gpu  = label_gpu[test_idx]
    
    test_model = xgb.XGBRegressor(n_estimators=solution[0],
                                  learning_rate=solution[1] * 0.01,
                                  max_depth=solution[2],
                                  min_child_weight=solution[3],
                                  gamma=solution[4] * 0.01,
                                  subsample=solution[5] * 0.01,
                                  colsample_bytree=solution[6] * 0.01,
                                  reg_lambda=solution[7] * 0.01,
                                  reg_alpha=solution[8] * 0.01,
                                  booster='gbtree',
                                  random_state=42,
                                  tree_method='hist', 
                                  device='cuda')
    
    test_model.fit(X_train_gpu, y_train_gpu)
    y_pred = test_model.predict(X_test_gpu)
    # 立即將預測結果轉換成 Cupy 陣列
    y_pred_gpu = cp.asarray(y_pred)
    mse = cp.mean((y_test_gpu - y_pred_gpu) ** 2).item()
    all_mse.append(mse)

final_mse_mean = np.mean(all_mse)
print("預測模型驗證MSE值:", final_mse_mean)


plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, color='blue', linestyle='--', marker='o', label='Best Solution Fitness')
plt.title('GA Fitness Evolution Over Generations', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()
