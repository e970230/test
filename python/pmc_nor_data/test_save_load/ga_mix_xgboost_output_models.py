import pygad
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import scipy.io
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import os

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

def fitness_func_method_two(ga_instance, solution, solution_idx):
    #判斷選擇使用全部特徵還是從中選取特徵
    data = init_data if num_params == 0 else init_data[:, solution[3:]]
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
    def train_and_evaluate(train_idx, test_idx):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    all_mse = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(train_idx, test_idx)      #將train_and_evaluate 函式延遲執行，使得它可以在 Parallel 中並行運行。
                                  for train_idx, test_idx in gkf.split(data, groups=group_labels))
    
    return -np.mean(all_mse)

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

#讀取檔案

init_data =scipy.io.loadmat('Final_train_nor_input_data_models_3.mat')['Final_input_data']       # 原始特徵數據(維度:樣本*特徵)
label = scipy.io.loadmat('Final_train_nor_output_data_models_3.mat')['Final_output_data']          # 特徵數據之對應標籤(維度:樣本*1)

test_input_data =  scipy.io.loadmat('Final_test_nor_input_data_models_3.mat')['Final_input_data']
test_output_data = scipy.io.loadmat('Final_test_nor_output_data_models_3.mat')['Final_output_data']

# 讀取特徵名稱
feature_names = scipy.io.loadmat('Final_Labels_name.mat')['Labels_name'].flatten()

# 定義分組標籤（假設已知每個樣本的組別）
group_labels = np.random.randint(0, 12, len(init_data))  # 這裡你可以使用實際的分組標籤

# 使用 GroupKFold 進行資料分割
gkf = GroupKFold(n_splits=6)     #設定 GroupKFold 交叉驗證模組

unique_numbers = np.unique(label)       #將標籤中不一樣處給區別出來，以後續處理使用

# 設定基因演算法參數
num_generations = 3                   #基因演算法疊代次數
num_parents_mating = 2                  #每代選多少個染色體進行交配
sol_per_pop = 3                        #染色體數量
num_params = 30                         #選擇的特徵數量
num_genes = 9 + num_params              #求解的數量


# 各個染色體範圍設置
gene_space = [
    {'low': 500, 'high': 1500},       # n_estimators
    {'low': 1, 'high': 50},         # learning_rate
    {'low': 1, 'high': 20},         # max_depth
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
                       random_seed = 42,
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

# 打印最佳超參數組合
print("最佳樹的數量 (n_estimators):", solution[0])
print("最佳學習率 (learning_rate):", solution[1] * 0.01)
print("最佳樹的最大深度 (max_depth):", solution[2])
print("最佳葉節點最小權重和 (min_child_weight):", solution[3])
print("最佳 gamma 值:", solution[4] * 0.01)
print("最佳 subsample 值:", solution[5] * 0.01)
print("最佳 colsample_bytree 值:", solution[6] * 0.01)
print("最佳 reg_lambda 值:", solution[7] * 0.01)
print("最佳 reg_alpha 值:", solution[8] * 0.01)

# 打印最佳選擇的特徵（降冪排列）
selected_features = np.sort(solution[9:])
selected_feature_names = feature_names[selected_features.astype(int)]
print("最佳選擇特徵:", selected_feature_names)

print("最佳解的適應度值:", solution_fitness)



# 儲存最佳解為 .mat 文件
scipy.io.savemat('best_solution_models_3.mat', {'solution': solution})

# 儲存最佳超參數和選擇特徵為 CSV 文件
model_name = 'models_3'  # 根據檔案名自動設定模型名稱
csv_filename = f'best_solution_{model_name}.csv'


solution_data = {
    'Parameter': [
        'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'gamma',
        'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha'
    ] + list(selected_feature_names),
    'Value': [
        solution[0], solution[1] * 0.01, solution[2], solution[3], solution[4] * 0.01,
        solution[5] * 0.01, solution[6] * 0.01, solution[7] * 0.01, solution[8] * 0.01
    ] + list(selected_features)
}

solution_df = pd.DataFrame(solution_data)
solution_df.to_csv(csv_filename, index=False)

#-------------------------------------------測試答案階段區域 two---------------------------------------
#判斷選擇使用全部特徵還是從中選取特徵
if num_params == 0:     #判斷是否選用基因演算法抽取特徵
    test_data = init_data    #不透過基因演算法選取特徵，則直接將整包data的所有特徵
else:
    test_data = test_input_data[:, solution[9:]]   # 擷取原數據共num_params個特徵，維度為(總數具樣本數*欲選擇特徵數量)

all_mse = []
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
                             random_state=42,
                             tree_method='hist', 
                             device='cuda')
test_model.fit(test_data, test_output_data)
y_pred = test_model.predict(test_data)
        

mse = mean_squared_error(test_output_data, y_pred)    #計算MSE值
all_mse.append(mse)                         #將當次MSE值記錄下來

# 使用 sklearn 的模組計算 MAPE
mape = mean_absolute_percentage_error(test_output_data, y_pred) * 100

print("預測模型驗證MSE值:", np.mean(all_mse))
print("MAPE", mape)

#------------------------------------儲存模型---------------------------------------------------

model_file = f'{model_name}_02.json'  # 模型檔案名稱
print("模型參數:", test_model.get_params())
test_model.save_model(model_file)
print(f"模型已成功保存為: {model_file}")

#------------------------------------------------------------------------------------------------

# 更新最佳超參數和選擇特徵的 CSV 文件，包含測試結果的 MSE 和 MAPE
solution_data.update({
    'Parameter': solution_data['Parameter'] + ['MSE', 'MAPE'],
    'Value': solution_data['Value'] + [np.mean(all_mse), mape]
})

solution_df = pd.DataFrame(solution_data)
solution_df.to_csv(csv_filename, index=False)

# 繪製適應度趨勢圖
plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, color='blue', linestyle='--', marker='o', label='Best Solution Fitness')
plt.title('GA Fitness Evolution Over Generations', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)

# 畫圖
plt.figure(figsize=(10, 6))

# 畫 y_test 的資料點
plt.scatter(range(len(test_output_data)), test_output_data, color='blue', label='True Values', marker='o')

# 畫 y_pred 的資料點
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', marker='o',s=20)

# 加標籤和標題
plt.title('True vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.show()
