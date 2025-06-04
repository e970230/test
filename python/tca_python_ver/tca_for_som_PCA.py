import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import pandas as pd

def perform_pca(train_data, pc_num=12):
    """
    執行標準化和PCA處理，並返回處理後的數據和主成分數量。
    
    :param train_data: 訓練數據
    :param variance_threshold: 累計方差佔比的閾值，用於選擇主成分數量
    :return: (處理後的PCA數據, 主成分數量, 標準化器, PCA模型)
    """
    # 初始化標準化器和PCA
    scaler = StandardScaler()
    pca_scaler = PCA()

    # 標準化數據
    train_scaled = scaler.fit_transform(train_data)

    # PCA數據
    train_pca = pca_scaler.fit_transform(train_scaled)

    print(train_pca.shape)
    print("每個主成分的方差佔比:", pca_scaler.explained_variance_ratio_)

    # 計算累計方差佔比
    pc = np.cumsum(pca_scaler.explained_variance_ratio_)
    print("累計主成分的方差佔比:", pc)


    # 提取指定數量的主成分
    train_pca = train_pca[:, :pc_num]

    return train_pca


# 指定要讀取的 CSV 檔案名稱
normal_file_name = "normal_test_1_A_feature_extraction.csv"  # 自行指定要讀取的檔案名稱
folder_path = r"C:\Users\user\Desktop\rowdata"

# 完整的 CSV 檔案路徑
file_path = os.path.join(folder_path, normal_file_name)

# 讀取 CSV
df_normal = pd.read_csv(file_path)
test_data_normal=np.array(df_normal)



# 計算每個樣本的尤拉距離（特徵平方和的平方根）
euclidean_distances_per_sample = np.sqrt(np.sum(test_data_normal**2, axis=1))

# 計算這些距離的標準差
euclidean_std = np.std(euclidean_distances_per_sample)

print("每個樣本的尤拉距離標準差:", euclidean_std)

test_data_normal = perform_pca(test_data_normal)

#test data
# 指定要讀取的 CSV 檔案名稱
abnormal_file_name = "abnormal_test_1_A_feature_extraction.csv"  # 自行指定要讀取的檔案名稱

# 完整的 CSV 檔案路徑
abnormal_file_path = os.path.join(folder_path, abnormal_file_name)


# 讀取 CSV
abnormal_df = pd.read_csv(abnormal_file_path)
test_data_abnormal=np.array(abnormal_df)

test_data_abnormal = perform_pca(test_data_abnormal)

normal_file_name_target = "normal_test_2_D_feature_extraction.csv"  # 自行指定要讀取的檔案名稱
abnormal_file_name_target = "abnormal_test_2_D_feature_extraction.csv"  # 自行指定要讀取的檔案名稱

# 讀取 CSV
df_normal_target = pd.read_csv(os.path.join(folder_path, normal_file_name_target))
test_data_normal_target=np.array(df_normal_target)

test_data_normal_target = perform_pca(test_data_normal_target)

# 讀取 CSV
abnormal_df_target = pd.read_csv(os.path.join(folder_path, abnormal_file_name_target))
test_data_abnormal_target=np.array(abnormal_df_target)

test_data_abnormal_target = perform_pca(test_data_abnormal_target)

# 資料整合
D_source = np.vstack((test_data_normal, test_data_abnormal))  # 將源域資料集x1和x2整合成一個矩陣
D_target = np.vstack((test_data_normal_target, test_data_abnormal_target))  # 將目標域資料集t1和t2整合成一個矩陣
ns = D_source.shape[0]          # 源域資料集數目
nt = D_target.shape[0]          # 目標域資料集數目
gamma = 0.1                     # RBF核函數超參數

# 核矩陣計算 (使用RBF)
def rbf_kernel(X, Y, gamma):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            diff = X[i, :] - Y[j, :]
            K[i, j] = np.exp(-gamma * np.sum(diff ** 2))
    return K

k_ss = rbf_kernel(D_source, D_source, gamma)
k_st = rbf_kernel(D_source, D_target, gamma)
k_ts = rbf_kernel(D_target, D_source, gamma)
k_tt = rbf_kernel(D_target, D_target, gamma)

# 合成完整的核矩陣 K
K = np.block([
    [k_ss, k_st],
    [k_ts, k_tt]
])  # 將上述四部分的核函數擺放在一起存成(kerneel matrix)核矩陣(大小為ns+nt*ns+nt)

# 計算M矩陣
Lambda = 0.1                                           # 計算M的超參數(懲罰因子)
u = np.ones((ns + nt, 1))                               # u為全都是1的轉置矩陣(大小為ns+nt*1)
H = np.eye(ns + nt) - (1 / (ns + nt)) * (u @ u.T)       # 計算H公式為(單位矩陣(大小為ns+nt*ns+nt)-(1/(ns+nt)*u*u轉置))

# L1計算過程由 ns個1/ns 和 nt個-1/nt 組成(大小為(ns+nt)*1的矩陣)

L1 = np.vstack((np.ones((ns, 1)) / ns, -np.ones((nt, 1)) / nt))
L = L1 @ L1.T

# M公式((K*L*K+Lambda*I).^-1)*K*H*K
M = np.linalg.inv(K @ L @ K + Lambda * np.eye(ns + nt)) @ (K @ H @ K)

# 特徵分解
D, V = np.linalg.eig(M)

# 把虛部去掉
D = np.real(D)
V = np.real(V)

# 特徵值由大到小排序
sorted_indices = np.argsort(-D)

sorted_D = D[sorted_indices]

pc=np.cumsum(sorted_D)
pc_true = pc / np.sum(sorted_D)

print("累計特徵值佔比:", pc_true)

# 設置中文支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用 Microsoft YaHei 字體
plt.rcParams['axes.unicode_minus'] = False  # 顯示負號

plt.figure(1)
plt.bar(np.arange(1,np.size(sorted_indices,0)+1),pc_true)
plt.xlabel('eigenvalue')
plt.ylabel('Cumulative percentage')
plt.legend()
plt.title('特徵值累計占比')
plt.show()

eigenvectors_sorted = V[:, sorted_indices]

# 取前三個特徵向量
W = eigenvectors_sorted[:, :15]

# 原始K投影到新特徵空間
answer = K @ W

tca_source_normal_data = answer[0:750,:]
tca_source_abnormal_data = answer[750:1500,:]
tca_target_normal_data = answer[1500:2250,:]
tca_target_abnormal_data = answer[2250:3000,:]

# 存成 CSV
pd.DataFrame(tca_source_normal_data).to_csv("tca_source_normal_data.csv", index=False)
pd.DataFrame(tca_source_abnormal_data).to_csv("tca_source_abnormal_data.csv", index=False)
pd.DataFrame(tca_target_normal_data).to_csv("tca_target_normal_data.csv", index=False)
pd.DataFrame(tca_target_abnormal_data).to_csv("tca_target_abnormal_data.csv", index=False)


print("   ")