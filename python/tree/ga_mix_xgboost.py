import pygad
import numpy as np
import xgboost as xgb
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
#    6. xgboost(1.7.6)
# 對於輸入數據之要求須注意維度必須是(樣本*特徵)的格式

# 最後修改時間:2024/7/8 

#-------------------------------------------定義副函式區域---------------------------------------
# 定義適應度函數
'''
def fitness_func_method_one(ga_instance, solution, solution_idx):
    data = feature_dataset[:, solution[3:]]                                 # 擷取原數據共num_params個特徵
    all_mse = []
    # 創建 XGBRegressor 模型
    model = xgb.XGBRegressor(n_estimators=int(solution[0]),          #將第一個解作為樹的數量
                             max_depth=int(solution[1]),             #將第二個解作為樹的最大深度
                             learning_rate=solution[2],              #將第三個解作為學習率
                             random_state=42)
    for fold in unique_numbers:
        
        X_train, X_test = data[np.where(label != fold)[0]], data[np.where(label == fold)[0]]
        y_train, y_test = label[np.where(label != fold)[0]], label[np.where(label == fold)[0]]

        try:
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return np.mean(scores),
        except Exception as e:
            print(f"Error evaluating solution {solution_idx}: {e}")
            return -np.inf,  # 返回一個極小值表示無效解決方案
'''
def fitness_func_method_two(ga_instance, solution, solution_idx):
    #判斷選擇使用全部特徵還是從中選取特徵
    if num_params == 0:     #判斷是否選用基因演算法抽取特徵
        data = init_data    #不透過基因演算法選取特徵，則直接將整包data的所有特徵
    else:
        data = init_data[:, solution[9:]]   # 擷取原數據共num_params個特徵，維度為(總數具樣本數*欲選擇特徵數量)
    all_mse = []
    #final_vr_answer = []
    # 創建 XGBRegressor 模型
    model = xgb.XGBRegressor(n_estimators= solution[0],             #將第一個解作為樹的數量
                             learning_rate= (solution[1]*0.01),     #將第三個解作為學習率
                             max_depth=solution[2],                 #將第二個解作為樹的最大深度
                             min_child_weight=solution[3],
                             gamma = solution[4]*0.01,
                             subsample = solution[5]*0.01,
                             colsample_bytree = solution[6]*0.01,
                             reg_lambda = solution[7]*0.01,
                             reg_alpha = solution[8]*0.01,
                             booster='gbtree',
                             random_state=42)
    for train_index, test_index in skf.split(data,label):
        X_train, X_test = data[train_index], data[test_index]               #將數據拆分成訓練數據和測試數據，並透過Kfold交叉驗證方式進行區分
        y_train, y_test = label[train_index], label[test_index]             #將標籤拆分成訓練標籤和測試標籤，並透過Kfold交叉驗證方式進行區分
        
        # 使用模型進行預測
        model.fit(X_train, y_train)                 #訓練模型
        y_pred = model.predict(X_test)              #預測解答

        # 計算預測答案和原始標籤之MSE值
        mse = mean_squared_error(y_test, y_pred)    #計算MSE值
        all_mse.append(mse)                         #將當次MSE值記錄下來
        '''
        verify_mse = {er_label: mean_squared_error(y_test[y_test == er_label], y_pred[y_test == er_label]) for er_label in unique_numbers}

        temporary_vr_answer = []  # 初始化為一個空列表，存儲每次的vr_answer

        for er_label, verify_mse in verify_mse.items():
            
            if verify_mse >= er_label:
                vr_answer = 1000
            else:
                vr_answer = verify_mse / er_label * 100
            
            vr_answer = verify_mse / er_label * 100
            temporary_vr_answer.append(vr_answer)  # 使用列表的append方法將vr_answer追加到final_vr_answer中
    
    final_vr_answer = np.max(temporary_vr_answer)
    '''
    #final_vr_answer.append(np.sum(temporary_vr_answer))  # 最後將列表轉換為numpy數組
    #final_vr_answer = np.array(final_vr_answer)  # 最後將列表轉換為numpy數組
    
    
    final_mse_mean = np.mean(all_mse, axis=0)       #將所有記錄下來的MSE值進行平均
    # 取負的MSE找其最大值
    return -final_mse_mean

#定義擷取特徵數量函數
def generate_dynamic_gene_space(num_params,init_data):
    Dimensions = np.shape(init_data)                #檢測數據的維度大小
    predefined_ranges = [{'low': 1, 'high': Dimensions[1]} for _ in range(num_params)] #根據設定所求選擇特徵之數量創建下限為1上限為矩陣特徵上限之字串設定
    return predefined_ranges

def generate_all_or_number(num_params,gene_space,init_data):
    gene_space_next = generate_dynamic_gene_space(num_params,init_data)
    if gene_space_next == []:
        final_gene_space = gene_space
    else:
        final_gene_space = gene_space + gene_space_next
    return final_gene_space


#額外顯示當次疊代當前fitness
def on_generation(ga_instance):
    print(f"第 {ga_instance.generations_completed} 代")
    print(f"最佳適應度值： {ga_instance.best_solutions_fitness[ga_instance.generations_completed-1]}")
    print("")


#-------------------------------------------主要運行程式區域---------------------------------------
'''
#讀取檔案
mat = scipy.io.loadmat('feature_dataset_heavy.mat')
feature_dataset = mat['feature_dataset']            #此原數據之輸入要求為樣本*特徵

'''
mat = scipy.io.loadmat('feature_dataset_top30.mat')

feature_dataset = mat['Data']

'''
# 讀取文件
with open('PCA_test_data.txt', 'r') as file:
    lines = file.readlines()

# 將每一行轉換為數字並存入矩陣
matrix = np.array([list(map(float, line.split())) for line in lines])
'''



init_data = feature_dataset[:, 1:]      # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]           # 擷取原數據的標籤，為原數據的第0列

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)     #設定sKfold交叉驗證模組(拆分的組數，再拆分前是否打亂，隨機性設定)

unique_numbers = np.unique(label)       #將標籤中不一樣處給區別出來，以後續處理使用

# 設定基因演算法參數
num_generations = 1000                   #基因演算法疊代次數
num_parents_mating = 15                  #每代選多少個染色體進行交配
sol_per_pop = 30                        #染色體數量
num_params = 0                         #選擇的特徵數量
num_genes = 9 + num_params              #求解的數量


# 各個染色體範圍設置
gene_space = [
    {'low': 100, 'high': 3000},       # n_estimators
    {'low': 1, 'high': 50},         # learning_rate
    {'low': 3, 'high': 20},         # max_depth
    {'low': 1, 'high': 10},         # min_child_weight
    {'low': 0, 'high': 50},         # gamma
    {'low': 50, 'high': 100},       # subsample
    {'low': 50, 'high': 100},       # colsample_bytree
    {'low': 0, 'high': 100},        # reg_lambda
    {'low': 0, 'high': 100},        # reg_alpha
]



final_gene_space = generate_all_or_number(num_params,gene_space,init_data)

# 基因演算法模型超參數細部設定
ga_instance = pygad.GA(
                       num_generations=num_generations,                #基因演算法疊代次數
                       num_parents_mating=num_parents_mating,          #每代選多少個染色體進行交配
                       fitness_func=fitness_func_method_two,           #定義適應度函數
                       sol_per_pop=sol_per_pop,                        #染色體數量
                       num_genes=num_genes,                            #求解的數量
                       gene_space=final_gene_space,                    #各個染色體範圍設置
                       gene_type=int,                                #每次疊代時基因演算法會以浮點數進行嘗試
                       parent_selection_type="rank",                   #選擇染色體方式依據排名來選擇
                       crossover_type="single_point",                  #單點交配
                       mutation_type="random",                         #隨機突變
                       mutation_probability=0.3,                       #突變率
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
'''
print("最佳樹的數量:", solution[0])
print("最佳樹的最大深度:", solution[1])
print("最佳葉節點最小權重和:", solution[2])
print("最佳選擇特徵:", np.sort(solution[3:]))
'''
print("最佳超參數組合:", solution)
print("最佳解的適應度值:", solution_fitness)


#-------------------------------------------測試答案階段區域 two---------------------------------------
#判斷選擇使用全部特徵還是從中選取特徵
if num_params == 0:     #判斷是否選用基因演算法抽取特徵
    test_data = init_data    #不透過基因演算法選取特徵，則直接將整包data的所有特徵
else:
    test_data = init_data[:, solution[9:]]   # 擷取原數據共num_params個特徵，維度為(總數具樣本數*欲選擇特徵數量)
label = feature_dataset[:, 0]
test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_mse = []
for train_index_test_ver, test_index_test_ver in test_skf.split(test_data,label):
        X_train, X_test = test_data[train_index_test_ver], test_data[test_index_test_ver]
        y_train, y_test = label[train_index_test_ver], label[test_index_test_ver]
        test_model = xgb.XGBRegressor(n_estimators= solution[0],             #將第一個解作為樹的數量
                             learning_rate= (solution[1]*0.01),     #將第三個解作為學習率
                             max_depth=solution[2],                 #將第二個解作為樹的最大深度
                             min_child_weight=solution[3],
                             gamma = solution[4]*0.01,
                             subsample = solution[5]*0.01,
                             colsample_bytree = solution[6]*0.01,
                             reg_lambda = solution[7]*0.01,
                             reg_alpha = solution[8]*0.01,
                             booster='gbtree',
                             random_state=42)
        test_model.fit(X_train, y_train)
        y_pred = test_model.predict(X_test)
        # 計算預測答案和原始標籤之MSE值
        mse = mean_squared_error(y_test, y_pred)
        all_mse.append(mse)


final_mse_mean = np.mean(all_mse, axis=0)

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
