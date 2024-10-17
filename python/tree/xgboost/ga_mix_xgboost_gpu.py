import pygad
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import scipy.io
import time
import matplotlib.pyplot as plt

#-------------------------------------------定義副函式區域---------------------------------------

def fitness_func_method_two(ga_instance, solution, solution_idx):
    if num_params == 0:     
        data = init_data    
    else:
        data = init_data[:, solution[9:]]   # 擷取特定的特徵

    all_mse = []
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
                             random_state=42,
                             tree_method='hist', 
                             device='cuda')  # 使用GPU

    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        model.fit(X_train, y_train)  # 訓練模型
        y_pred = model.predict(X_test)  # 預測

        mse = mean_squared_error(y_test, y_pred)  # 計算MSE
        all_mse.append(mse)

    final_mse_mean = np.mean(all_mse, axis=0)  # MSE平均值
    return -final_mse_mean  # 返回負的MSE找最小值

#-------------------------------------------主要運行程式區域---------------------------------------

mat = scipy.io.loadmat('feature_dataset_top30.mat')
feature_dataset = mat['Data']

init_data = feature_dataset[:, 1:]  # 擷取特徵
label = feature_dataset[:, 0]  # 擷取標籤

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

unique_numbers = np.unique(label)

num_generations = 1000
num_parents_mating = 25
sol_per_pop = 50
num_params = 0
num_genes = 9 + num_params

gene_space = [
    {'low': 500, 'high': 1500},       # n_estimators
    {'low': 1, 'high': 50},           # learning_rate
    {'low': 1, 'high': 20},           # max_depth
    {'low': 1, 'high': 10},           # min_child_weight
    {'low': 0, 'high': 50},           # gamma
    {'low': 50, 'high': 100},         # subsample
    {'low': 50, 'high': 100},         # colsample_bytree
    {'low': 0, 'high': 100},          # reg_lambda
    {'low': 0, 'high': 100},          # reg_alpha
]

def generate_dynamic_gene_space(num_params, init_data):
    Dimensions = np.shape(init_data)
    predefined_ranges = [{'low': 1, 'high': Dimensions[1]} for _ in range(num_params)]
    return predefined_ranges

def generate_all_or_number(num_params, gene_space, init_data):
    gene_space_next = generate_dynamic_gene_space(num_params, init_data)
    if gene_space_next == []:
        return gene_space
    else:
        return gene_space + gene_space_next

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
    on_generation=lambda ga: print(f"第 {ga.generations_completed} 代\n最佳適應度值： {ga.best_solutions_fitness[ga.generations_completed-1]}\n"),
)

start_time = time.time()
ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed Time: {elapsed_time:.2f} seconds")
print(f"最佳超參數組合: {solution}")
print(f"最佳解的適應度值: {solution_fitness}")

# 測試最佳解的模型
if num_params == 0:
    test_data = init_data
else:
    test_data = init_data[:, solution[9:]]

all_mse = []
for train_index_test_ver, test_index_test_ver in skf.split(test_data, label):
    X_train, X_test = test_data[train_index_test_ver], test_data[test_index_test_ver]
    y_train, y_test = label[train_index_test_ver], label[test_index_test_ver]

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
                                  device='cuda')  # 使用GPU

    test_model.fit(X_train, y_train)
    y_pred = test_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    all_mse.append(mse)

final_mse_mean = np.mean(all_mse, axis=0)

# 輸出結果
print(f"預測模型驗證MSE值: {final_mse_mean}")

# 繪製適應度趨勢圖
plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, color='blue', linestyle='--', marker='o', label='Best Solution Fitness')
plt.title('GA Fitness Evolution Over Generations', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()
