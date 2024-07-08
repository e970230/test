import pygad
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
import scipy.io
import time
import matplotlib.pyplot as plt

# 定義適應度函數
def fitness_func(ga_instance, solution, solution_idx):
    data = feature_dataset[:, solution[3:]]        # 擷取原數據共num_params個特徵，
    all_mse = []
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]               #將數據拆分成訓練數據和測試數據，並透過
        y_train, y_test = label[train_index], label[test_index]
        
        # 創建 ExtraTreesRegressor 模型
        model = ExtraTreesRegressor(n_estimators=int(solution[0]),          #將第一個解作為樹的數量
                                    max_features=int(solution[1]),          #將第二個解作為分裂時考慮的最大特徵數
                                    min_samples_split=int(solution[2]),     #將第三個解作為葉節點最小樣本數
                                    random_state=42)

        # 使用模型進行預測
        model.fit(X_train, y_train)         #訓練模型
        y_pred = model.predict(X_test)      #預測解答

        # 計算預測答案和原始標籤之MSE值
        mse = mean_squared_error(y_test, y_pred)    #計算MSE值
        all_mse.append(mse)                         #將當次MSE值記錄下來
    
    final_mse_mean = np.mean(all_mse, axis=0)       #將所有記錄下來的MSE值進行平均
    # 取負的MSE找其最大值
    return -final_mse_mean

#定義擷取特徵數量函數
def generate_dynamic_gene_space(num_params,init_data):
    Dimensions = np.shape(init_data)                #檢測數據的維度大小
    predefined_ranges = []                          #創建關於選擇特徵的基因上下限設定空矩陣
    for _ in range(num_params):
        predefined_ranges.append({'low': 1, 'high': Dimensions[1]})
    return predefined_ranges

#額外顯示當次疊代當前fitness
def on_generation(ga_instance):
    print(f"第 {ga_instance.generations_completed} 代")
    print(f"最佳適應度值： {ga_instance.best_solution()[1]}")
    print("")


mat = scipy.io.loadmat('feature_dataset_heavy.mat')

feature_dataset = mat['feature_dataset']


init_data = feature_dataset[:, 1:]  # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]  # 擷取原數據的標籤，為原數據的第0列

kf = KFold(n_splits=5, shuffle=True, random_state=None)


start_time = time.time()


# 設定基因演算法參數
num_generations = 3       #基因演算法疊代次數
num_parents_mating = 5      #每代選多少個染色體進行交配
sol_per_pop = 20            #染色體數量
num_params = 30             #選擇的特徵數量
num_genes = 3 + num_params               #求解的數量


# 各個染色體範圍設置
gene_space = [
    {'low': 10, 'high': 300},  # n_estimators
    {'low': 1, 'high': 30},    # max_features
    {'low': 2, 'high': 20}     # min_samples_split
]

gene_feature_space = generate_dynamic_gene_space(num_params,init_data)

final_gene_space = gene_space + gene_feature_space

# 基因演算法模型超參數細部設定
ga_instance = pygad.GA(
                       num_generations=num_generations,                #基因演算法疊代次數
                       num_parents_mating=num_parents_mating,          #每代選多少個染色體進行交配
                       fitness_func=fitness_func,                      #定義適應度函數
                       sol_per_pop=sol_per_pop,                        #染色體數量
                       num_genes=num_genes,                            #求解的數量
                       gene_space=final_gene_space,                    #各個染色體範圍設置
                       gene_type=int,                                  #每次疊代時基因演算法會以整數進行嘗試
                       parent_selection_type="rank",                   #選擇染色體方式依據排名來選擇
                       crossover_type="single_point",                  #單點交配
                       mutation_type="random",                         #隨機突變
                       mutation_probability=0.3,                       #突變率
                       on_generation=on_generation,                    #每次疊代資訊顯示
                       save_solutions=False                             #儲存每次疊代解答
)

# 執行基因演算法
ga_instance.run()

# 取得最佳解
solution, solution_fitness, solution_idx = ga_instance.best_solution()

end_time = time.time()
elapsed_time = end_time - start_time                #顯示運行時間
print("Elapsed Time: %.2f seconds" % elapsed_time)
print("最佳樹的數量:", solution[0])
print("最佳分裂時考慮的最大特徵數:", solution[1])
print("最佳葉節點最小樣本數:", solution[2])
print("最佳選擇特徵:", solution[3:])
print("最佳解的適應度值:", solution_fitness)




#-------------------------------------------測試答案階段---------------------------------------
#KFold隨機種子設定為21時
test_data = feature_dataset[:, solution[3:]]
label = feature_dataset[:, 0]
all_mse = []
test_kf = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index_test_ver, test_index_test_ver in test_kf.split(test_data):
        X_train, X_test = test_data[train_index_test_ver], test_data[test_index_test_ver]
        y_train, y_test = label[train_index_test_ver], label[test_index_test_ver]
        test_model = ExtraTreesRegressor(n_estimators=solution[0],
                                    max_features=solution[1],
                                    min_samples_split=solution[2],
                                    random_state=42)
        test_model.fit(X_train, y_train)
        y_pred = test_model.predict(X_test)
        # 計算預測答案和原始標籤之MSE值
        mse = mean_squared_error(y_test, y_pred)
        all_mse.append(mse)

final_mse_mean = np.mean(all_mse, axis=0)

print("預測模型驗證MSE值:",final_mse_mean)


# 繪製適應度趨勢圖

plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, color='blue', linestyle='--', marker='o', label='Best Solution Fitness')
plt.title('GA Fitness Evolution Over Generations', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()