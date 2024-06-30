import pygad
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
import scipy.io
import time


mat = scipy.io.loadmat('feature_dataset_heavy.mat')

feature_dataset = mat['feature_dataset']


data = feature_dataset[:, 1:]  # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]  # 擷取原數據的標籤，為原數據的第0列


#將數據百分之20做為測試集百分之80做為訓練集
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20)


start_time = time.time()

# 定義適應度函數
def fitness_func(ga_instance, solution, solution_idx):
    n_estimators = int(solution[0])
    max_features = int(solution[1])
    min_samples_split = int(solution[2])

    # 創建 ExtraTreesRegressor 模型
    model = ExtraTreesRegressor(n_estimators=n_estimators,
                                max_features=max_features,
                                min_samples_split=min_samples_split,
                                random_state=42)

    # 使用交叉驗證計算 MAE
    #scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
    #mean_score = np.mean(scores)
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)

    # 計算預測答案和原始標籤之MSE和RMSE值
    mse = mean_squared_error(y_test, y_pred)

    # 由於 cross_val_score 返回負的 MAE，我們需要取其相反數
    return -mse


def on_generation(ga_instance):
    print(f"第 {ga_instance.generations_completed} 代")
    print(f"最佳適應度值： {ga_instance.best_solution()[1]}")
    print("")

# 設定基因演算法參數
num_generations = 1000
num_parents_mating = 5
sol_per_pop = 20
num_genes = 3

# 基因範圍設置
gene_space = [
    {'low': 10, 'high': 300},  # n_estimators
    {'low': 1, 'high': 30},    # max_features
    {'low': 2, 'high': 20}     # min_samples_split
]

# 初始化基因演算法實例
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=gene_space,
    gene_type=int,
    parent_selection_type="rank",
    crossover_type="single_point",
    mutation_type="random",
    mutation_probability=0.3,
    on_generation=on_generation
)

# 執行基因演算法
ga_instance.run()

# 取得最佳解
solution, solution_fitness, solution_idx = ga_instance.best_solution()

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed Time: %.2f seconds" % elapsed_time)
print("最佳解:", solution)
print("最佳解的適應度值:", solution_fitness)

# 繪製適應度趨勢圖
ga_instance.plot_fitness()
