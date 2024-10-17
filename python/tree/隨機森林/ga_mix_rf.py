import pygad
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import scipy.io
import time
import matplotlib.pyplot as plt

# 此為透過使用pygad的基因演算法對xgboost的XGBRegressor進行超參數優化來使預測結果更精準
# 在使用時須先注意若要運行此程式需先import下列模組才能使用，當下測試模組版本和python版本會記錄下來做為參考
#    測試運行當下python版本為 3.11.2
#    1. pygad(3.3.1)
#    2. numpy(1.26.4)
#    3. sklearn(1.5.0)
#    4. scipy(1.14.0)
#    5. matplotlib(3.9.0)
# 對於輸入數據之要求須注意維度必須是(樣本*特徵)的格式


#-------------------------------------------定義costfunction---------------------------------------

def fitness_func_method_two(ga_instance, solution, solution_idx):
    if num_params == 0:     
        data = init_data                    #使用原始所有特徵的data
    else:
        data = init_data[:, solution[3:]]   # 擷取特定的特徵

    all_mse = []
    model = RandomForestRegressor(n_estimators=solution[0],         # 樹的數量
                                  max_depth=solution[1],            # 最大深度
                                  min_samples_split=solution[2],    # 最小樣本數
                                  random_state=42,                  # 固定隨機種子
                                  n_jobs=-1)                        # 使用所有CPU核心

    for train_index, test_index in skf.split(data, label):          #透過K-fold交叉驗證將數據拆分成訓練數據和驗證數據
        X_train, X_test = data[train_index], data[test_index]       
        y_train, y_test = label[train_index], label[test_index]     

        model.fit(X_train, y_train)  # 訓練模型
        y_pred = model.predict(X_test)  # 預測

        mse = mean_squared_error(y_test, y_pred)  # 計算MSE
        all_mse.append(mse)

    final_mse_mean = np.mean(all_mse, axis=0)  # MSE平均值
    return -final_mse_mean  # 返回負的MSE找最小值

#-------------------------------------------主要運行程式區域---------------------------------------


#-------------------匯入數據區域-------------------
mat = scipy.io.loadmat('feature_dataset_top30.mat')
feature_dataset = mat['Data']

init_data = feature_dataset[:, 1:]      # 原始特徵數據(維度:樣本*特徵)
label = feature_dataset[:, 0]           # 特徵數據之對應標籤(維度:樣本*1)


#--------------------------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)        #設定sKfold交叉驗證模組(拆分的組數，再拆分前是否打亂，隨機性設定)

unique_numbers = np.unique(label)           #抓取不同標籤種類的數量

num_generations = 10                #疊代次數
num_parents_mating = 2              #交配染色體的數量
sol_per_pop = 5                     #染色體的數量
num_params = 0                      #想抓取的特徵數量(若不想抓取則輸入0，意即不抓取特徵使用當前所有特徵)
num_genes = 3 + num_params          #欲求解數量；欲求解數量 = RF欲求超參數數量 + 欲求特徵數量 (ex:若我想使用RF並從原始數據中抓取30個特徵，則變數數量則為 3 + 30 = 33)
                                    #RF的3個欲求變數分別為 : n_estimators = 樹的數量；max_depth = 樹的最大深度 ； min_samples_split = 葉節點最小樣本數。


# 隨機森林的超參數範圍
gene_space = [
    {'low': 100, 'high': 1500},   # n_estimators 樹的數量
    {'low': 1, 'high': 30},       # max_depth 樹的最大深度
    {'low': 5, 'high': 10}        # min_samples_split 葉節點最小樣本數。
]


#定義擷取特徵數量函數
def generate_dynamic_gene_space(num_params, init_data):
    Dimensions = np.shape(init_data)                                                        #檢測數據的維度大小
    predefined_ranges = [{'low': 1, 'high': Dimensions[1]} for _ in range(num_params)]      #根據設定所求選擇特徵之數量創建下限為1上限為矩陣特徵上限之字串設定
    return predefined_ranges


#依據是否有選擇特徵來進行判斷的副函式
def generate_all_or_number(num_params, gene_space, init_data):
    gene_space_next = generate_dynamic_gene_space(num_params, init_data)
    if gene_space_next == []:                   #若沒有透過基因演算法選取特徵時
        return gene_space                       #則以先前的求解範圍設置作為最終設置
    else:
        return gene_space + gene_space_next     #若使用基因演算法選擇特徵時，則在原求解範圍後增加新的特徵求解範圍配置(數量為num_params個下限為1上限為特徵數量最大值的設置)

final_gene_space = generate_all_or_number(num_params, gene_space, init_data)    #設定最終的欲求解的範圍，若有啟動GA抽取特徵則在此一併處理

ga_instance = pygad.GA(
    num_generations=num_generations,                #基因演算法疊代次數
                       num_parents_mating=num_parents_mating,          #每代選多少個染色體進行交配
                       fitness_func=fitness_func_method_two,           #定義適應度函數
                       sol_per_pop=sol_per_pop,                        #染色體數量
                       num_genes=num_genes,                            #求解的數量
                       gene_space=final_gene_space,                    #各個染色體範圍設置
                       gene_type=int,                                  #每次疊代時基因演算法會以整數進行嘗試
                       parent_selection_type="rank",                   #選擇染色體方式依據排名來選擇
                       crossover_type="single_point",                  #單點交配
                       mutation_type="random",                         #隨機突變
                       mutation_probability=0.4,                       #突變率
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


#---------------------------------驗證模型階段---------------------------------

# 測試最佳解的模型
if num_params == 0:
    test_data = init_data                       #若沒有使用GA抽取特徵，則直接使用原始數據
else:
    test_data = init_data[:, solution[3:]]      #若有使用GA抽取特徵，則使用抽取完的特徵數據

#---------------------------------驗證模型建立---------------------------------

all_mse = []
for train_index_test_ver, test_index_test_ver in skf.split(test_data, label):
    X_train, X_test = test_data[train_index_test_ver], test_data[test_index_test_ver]
    y_train, y_test = label[train_index_test_ver], label[test_index_test_ver]

    test_model = RandomForestRegressor(n_estimators=solution[0],        # 樹的數量
                                       max_depth=solution[1],           # 最大深度
                                       min_samples_split=solution[2],   # 最小樣本數
                                       random_state=42,
                                       n_jobs=-1)

    test_model.fit(X_train, y_train)
    y_pred = test_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    all_mse.append(mse)

final_mse_mean = np.mean(all_mse, axis=0)

# 輸出結果
print(f"預測模型驗證MSE值: {final_mse_mean}")

verify_mse = {er_label: mean_squared_error(y_test[y_test == er_label], y_pred[y_test == er_label]) for er_label in unique_numbers}

# 輸出每個標籤的mse值
for er_label, verify_mse in verify_mse.items():
    if verify_mse>=er_label:
        er_answer = " 誤差率過大 "
    else:
        er_answer = f" 誤差率百分之{verify_mse/er_label*100} "
    print(f"預壓力 {er_label} 的MSE值為 {verify_mse:.2f} {er_answer}")

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
