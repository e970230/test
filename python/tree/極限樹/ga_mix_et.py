import pygad
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
import scipy.io
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# 此為透過使用pygad的基因演算法對sklearn的ExtraTreesRegressor進行超參數優化來使預測結果更精準
# 在使用時須先注意若要運行此程式需先import下列模組才能使用，當下測試模組版本和python版本會記錄下來做為參考
#    測試運行當下python版本為 3.11.2
#    1. pygad(3.3.1)
#    2. numpy(1.26.4)
#    3. sklearn(1.5.0)
#    4. scipy(1.14.0)
#    5. matplotlib(3.9.0)
#    6. joblib(1.4.2)
# 對於輸入數據之要求須注意維度必須是(樣本*特徵)的格式


# 最後修改時間:2024/10/16 

#-------------------------------------------定義副函式區域---------------------------------------
# 定義適應度函數

def fitness_func_method(ga_instance, solution, solution_idx):
    data = init_data if num_params == 0 else init_data[:, solution[3:]]

    

    # 創建 ExtraTreesRegressor 模型
    model = ExtraTreesRegressor(n_estimators=(solution[0]),          #將第一個解作為樹的數量
                                max_features=(solution[1]),          #將第二個解作為分裂時考慮的最大特徵數
                                min_samples_split=(solution[2]),     #將第三個解作為葉節點最小樣本數
                                random_state=42)
    def train_and_evaluate(train_idx, test_idx):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    all_mse = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(train_idx, test_idx)      #將train_and_evaluate 函式延遲執行，使得它可以在 Parallel 中並行運行。
                                  for train_idx, test_idx in skf.split(data, label))
    
    return -np.mean(all_mse)

#定義擷取特徵數量函數
def generate_dynamic_gene_space(num_params,init_data):
    Dimensions = np.shape(init_data)                #檢測數據的維度大小
    predefined_ranges = [{'low': 1, 'high': Dimensions[1]} for _ in range(num_params)] #根據設定所求選擇特徵之數量創建下限為1上限為矩陣特徵上限之字串設定
    return predefined_ranges


#依據是否有選擇特徵來進行判斷的副函式
def generate_all_or_number(num_params,gene_space,init_data):
    gene_space_next = generate_dynamic_gene_space(num_params,init_data)
    if gene_space_next == []:           #若沒有透過基因演算法選取特徵時
        final_gene_space = gene_space   #則以先前的求解範圍設置作為最終設置
    else:
        final_gene_space = gene_space + gene_space_next     #若使用基因演算法選擇特徵時，則在原求解範圍後增加新的特徵求解範圍配置(數量為num_params個下限為1上限為特徵數量最大值的設置)
    return final_gene_space




#-------------------------------------------主要運行程式區域---------------------------------------

#透過LTFRM選取最佳前30特徵的預壓力資料

mat = scipy.io.loadmat('feature_dataset_top30.mat')

feature_dataset = mat['Data']

init_data = feature_dataset[:, 1:]      # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]           # 擷取原數據的標籤，為原數據的第0列

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)     #設定sKfold交叉驗證模組(拆分的組數，再拆分前是否打亂，隨機性設定)

unique_numbers = np.unique(label)       #將標籤中不相同處給區別出來，以後續處理使用

# 設定基因演算法參數
num_generations = 10                   #基因演算法疊代次數
num_parents_mating = 2                 #每代選多少個染色體進行交配
sol_per_pop = 5                        #染色體數量
num_params = 0                          #選擇的特徵數量，若不使用基因演算法選取特徵則此處請輸入0
num_genes = 3 + num_params              #求解的數量


# 各個求解範圍設置
gene_space = [
    {'low': 100, 'high': 600},  # n_estimators(樹的數量)
    {'low': 1, 'high': 40},    # max_features(分裂時考慮的最大特徵數)
    {'low': 2, 'high': 30}     # min_samples_split(葉節點最小樣本數)
]

# 設置選擇特徵的基因範圍，若想從50個特徵則會創建一個50個{'low': 1, 'high': data的特徵上限}+前面設置的染色體範圍的配置
final_gene_space = generate_all_or_number(num_params,gene_space,init_data)

# 基因演算法模型超參數細部設定
ga_instance = pygad.GA(
                       num_generations=num_generations,                #基因演算法疊代次數
                       num_parents_mating=num_parents_mating,          #每代選多少個染色體進行交配
                       fitness_func=fitness_func_method,               #定義適應度函數
                       sol_per_pop=sol_per_pop,                        #染色體數量
                       num_genes=num_genes,                            #求解的數量
                       gene_space=final_gene_space,                    #各個染色體範圍設置
                       gene_type=int,                                  #每次疊代時基因演算法會以整數進行嘗試
                       parent_selection_type="rank",                   #選擇染色體方式依據排名來選擇
                       crossover_type="single_point",                  #單點交配
                       mutation_type="random",                         #隨機突變
                       mutation_probability=0.4,                       #突變率
                       random_seed=42,                                 #隨機種子設定
                       save_solutions=False,                            #儲存每次疊代解答之設定
                       on_generation=lambda ga: print(f"第 {ga.generations_completed} 代\n最佳適應度值： {ga.best_solutions_fitness[ga.generations_completed-1]}\n"),#每次疊代資訊顯示
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
print("最佳選擇特徵:", np.sort(solution[3:]))
print("最佳解的適應度值:", solution_fitness)




#-------------------------------------------測試答案階段區域---------------------------------------
#重新根據最佳解答再次建立模型進行判斷
final_vr_answer = np.empty((0, len(unique_numbers)))
#判斷選擇使用全部特徵還是從中選取特徵
if num_params == 0:         #判斷是否選用基因演算法抽取特徵
    test_data = init_data   #不透過基因演算法選取特徵，則直接將整包data的所有特徵
else:
    test_data = init_data[:, solution[3:]]   # 擷取原數據共num_params個特徵，維度為(總數具樣本數*欲選擇特徵數量)

label = feature_dataset[:, 0]
test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_mse = []
for train_index_test_ver, test_index_test_ver in test_skf.split(test_data,label):
        X_train, X_test = test_data[train_index_test_ver], test_data[test_index_test_ver]       #將數據拆分成訓練數據和測試數據，並透過Kfold交叉驗證方式進行區分
        y_train, y_test = label[train_index_test_ver], label[test_index_test_ver]               #將數據拆分成訓練標籤和測試標籤，並透過Kfold交叉驗證方式進行區分
        test_model = ExtraTreesRegressor(n_estimators=(solution[0]),            #將第一個解作為樹的數量
                                max_features=(solution[1]),                     #將第二個解作為分裂時考慮的最大特徵數
                                min_samples_split=(solution[2]),                #將第三個解作為葉節點最小樣本數
                                random_state=42)
        
        
        test_model.fit(X_train, y_train)        #訓練模型
        y_pred = test_model.predict(X_test)     #預測解答


        # 計算每種標籤預測答案和原始標籤之MSE值
        verify_mse = {er_label: mean_squared_error(y_test[y_test == er_label], y_pred[y_test == er_label]) for er_label in unique_numbers}
        temporary_vr_answer = []  # 初始化為一個空列表，存儲每次的vr_answer

        #將每種標籤各自的MSE值進行判斷
        for er_label, verify_mse in verify_mse.items():
            if verify_mse >= er_label:  #若MSE過大則以一個大數去代替
                vr_answer = 1000
            else:
                vr_answer = verify_mse / er_label * 100 #若MSE正常則計算其錯誤百分比

            temporary_vr_answer.append(vr_answer)  # 使用列表的append方法將vr_answer追加到final_vr_answer中
        

        temporary_vr_answer = np.array(temporary_vr_answer).reshape(1, -1)  # 將其轉換為一行
        final_vr_answer = np.vstack((final_vr_answer, temporary_vr_answer)) # 垂直堆疊

        # 計算預測答案和原始標籤之MSE值
        mse = mean_squared_error(y_test, y_pred)    #計算MSE值
        all_mse.append(mse)                         #將當次MSE值記錄下來
    


row_means = np.mean(final_vr_answer, axis=0)    #取每個fold中同種標籤種類的MSE
final_mse_mean = np.mean(all_mse, axis=0)       #將所有記錄下來的MSE值進行平均

# 輸出每個標籤的mse值
# 假設 unique_numbers 是你的標籤對應的數據
for er_label, row_mean in zip(unique_numbers, row_means):
    if row_mean >= er_label:
        er_answer = "誤差率過大"
    else:
        er_answer = f"誤差率百分之 {row_mean / er_label * 100:.2f}"

    print(f"預壓力 {er_label} 的MSE值為 {row_mean:.2f} {er_answer}")

print("預測模型驗證MSE值:", final_mse_mean)


# 繪製適應度趨勢圖

plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, color='blue', linestyle='--', marker='o', label='Best Solution Fitness')
plt.title('GA Fitness Evolution Over Generations', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()