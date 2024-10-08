import pygad
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
import scipy.io
import time
import matplotlib.pyplot as plt

# 此為透過使用pygad的基因演算法對sklearn的ExtraTreesRegressor進行超參數優化來使預測結果更精準
# 在使用時須先注意若要運行此程式需先import下列模組才能使用，當下測試模組版本和python版本會記錄下來做為參考
# 此為kfold驗證方法1: 再拆分資料時故意將某一預標籤整體作為測試資料不讓他測試，藉此對整體模型進行壓力測試
#    測試運行當下python版本為 3.11.2
#    1. pygad(3.3.1)
#    2. numpy(1.26.4)
#    3. sklearn(1.5.0)
#    4. scipy(1.14.0)
#    5. matplotlib(3.9.0)
# 對於輸入數據之要求須注意維度必須是(樣本*特徵)的格式


# 最後修改時間:2024/7/8 

#-------------------------------------------定義副函式區域---------------------------------------
# 定義適應度函數


def fitness_func(ga_instance, solution, solution_idx):
    data = feature_dataset[:, solution[3:]]                                 # 擷取原數據共num_params個特徵，
    all_mse = []
    for fold in unique_numbers:
        
        X_train, X_test = data[np.where(label != fold)[0]], data[np.where(label == fold)[0]]
        y_train, y_test = label[np.where(label != fold)[0]], label[np.where(label == fold)[0]]

        # 創建 ExtraTreesRegressor 模型
        model = ExtraTreesRegressor(n_estimators=int(solution[0]),          #將第一個解作為樹的數量
                                    max_features=int(solution[1]),          #將第二個解作為分裂時考慮的最大特徵數
                                    min_samples_split=int(solution[2]),     #將第三個解作為葉節點最小樣本數
                                    random_state=42)

        # 使用模型進行預測
        model.fit(X_train, y_train)                 #訓練模型
        y_pred = model.predict(X_test)              #預測解答

        # 計算預測答案和原始標籤之MSE值
        mse = mean_squared_error(y_test, y_pred)    #計算MSE值
        all_mse.append(mse)                         #將當次MSE值記錄下來
    
    final_mse_mean = np.mean(all_mse, axis=0)       #將所有記錄下來的MSE值進行平均
    # 取負的MSE找其最大值
    return -final_mse_mean

#定義擷取特徵數量函數
def generate_dynamic_gene_space(num_params,init_data):
    Dimensions = np.shape(init_data)                #檢測數據的維度大小
    predefined_ranges = []                          #創建關於選擇特徵的基因上下限設定之矩陣
    for _ in range(num_params):
        predefined_ranges.append({'low': 1, 'high': Dimensions[1]})     #根據設定所求選擇特徵之數量創建下限為1上限為矩陣特徵上限之字串設定
    return predefined_ranges

#額外顯示當次疊代當前fitness
def on_generation(ga_instance):
    print(f"第 {ga_instance.generations_completed} 代")
    print(f"最佳適應度值： {ga_instance.best_solution()[1]}")
    print("")


#-------------------------------------------主要運行程式區域---------------------------------------

#讀取檔案
mat = scipy.io.loadmat('feature_dataset_heavy.mat')
feature_dataset = mat['feature_dataset']            #此原數據之輸入要求為樣本*特徵


init_data = feature_dataset[:, 1:]      # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]           # 擷取原數據的標籤，為原數據的第0列

unique_numbers = np.unique(label)       #將標籤中不一樣處給區別出來，以後續處理使用

# 設定基因演算法參數
num_generations = 3                   #基因演算法疊代次數
num_parents_mating = 10                  #每代選多少個染色體進行交配
sol_per_pop = 20                        #染色體數量
num_params = 30                         #選擇的特徵數量
num_genes = 3 + num_params              #求解的數量


# 各個染色體範圍設置
gene_space = [
    {'low': 10, 'high': 300},  # n_estimators
    {'low': 1, 'high': 40},    # max_features
    {'low': 2, 'high': 30}     # min_samples_split
]

                                               
gene_feature_space = generate_dynamic_gene_space(num_params,init_data)      #透過自訂義函數創建關於選擇特徵的基因上下限設定之矩陣
#輸入參數(想選擇的特徵數量，原數據本身==>用於觀測原數據之特徵數量上限)
final_gene_space = gene_space + gene_feature_space                          #將設定基礎極限樹之模型上下限和選擇特徵的基因上下限統整再一起

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
                       mutation_probability=0.4,                       #突變率
                       on_generation=on_generation,                    #每次疊代資訊顯示
                       save_solutions=False                            #儲存每次疊代解答之設定
)

start_time = time.time()

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




#-------------------------------------------測試答案階段區域---------------------------------------
'''
test_data = feature_dataset[:, solution[3:]]
label = feature_dataset[:, 0]
all_mse = []
test_skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
for train_index_test_ver, test_index_test_ver in test_skf.split(test_data,label):
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
'''

# 繪製適應度趨勢圖

plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, color='blue', linestyle='--', marker='o', label='Best Solution Fitness')
plt.title('GA Fitness Evolution Over Generations', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()