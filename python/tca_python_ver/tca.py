import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

# 讀取資料
data = scipy.io.loadmat('TCA_exercise_dataset.mat')
x1 = data['x1'] 
x2 = data['x2']
t1 = data['t1']
t2 = data['t2']

# 資料整合
D_source = np.vstack((x1, x2))  # 將源域資料集x1和x2整合成一個矩陣
D_target = np.vstack((t1, t2))  # 將目標域資料集t1和t2整合成一個矩陣
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
eigenvectors_sorted = V[:, sorted_indices]

# 取前三個特徵向量
W = eigenvectors_sorted[:, :3]

# 原始K投影到新特徵空間
answer = K @ W


# 設定支援中文
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.set_title("將K投影在前三特徵向量")


ax.scatter(answer[0:100,0], answer[0:100,1], answer[0:100,2], marker='+', label='x1')
ax.scatter(answer[100:200,0], answer[100:200,1], answer[100:200,2], marker='o', label='x2')
ax.scatter(answer[200:220,0], answer[200:220,1], answer[200:220,2], marker='+', label='t1')
ax.scatter(answer[220:240,0], answer[220:240,1], answer[220:240,2], marker='o', label='t2')

ax.legend()
plt.show()


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("前三特徵向量")
ax2.scatter(W[0:100,0], W[0:100,1], W[0:100,2], marker='+', label='x1')
ax2.scatter(W[100:200,0], W[100:200,1], W[100:200,2], marker='o', label='x2')
ax2.scatter(W[200:220,0], W[200:220,1], W[200:220,2], marker='+', label='t1')
ax2.scatter(W[220:240,0], W[220:240,1], W[220:240,2], marker='o', label='t2')
ax2.legend()
plt.show()
