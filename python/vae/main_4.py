import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# 檢查設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. 載入資料與預處理
# ---------------------------

mat_data = scipy.io.loadmat('all_input.mat')
x_train = mat_data['all_input']  # 載入變數

# 讀取 CSV 與 pickle 並提取特徵索引
best_feature_models_1_F_10000 = pd.read_csv("best_solution_models_1_F_10000.csv")
model_1 = pickle.load(open('models_1_F_10000.pkl', "rb"))
feature_indices = best_feature_models_1_F_10000.iloc[9:-2, 1].astype(int).values
# 依據索引選擇特徵 
x_train = x_train[:, feature_indices]

train_predictions = model_1.predict(x_train)

# 正規化到 0~1
min_val = np.min(x_train)
max_val = np.max(x_train)
x_train_scaled = (x_train - min_val) / (max_val - min_val)

# 將數據轉成 torch.Tensor
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
print("x_train_tensor shape:", x_train_tensor.shape)

# 建立 dataset 與 dataloader
dataset = TensorDataset(x_train_tensor)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------------------
# 2. 定義三層 VAE 模型
# ---------------------------

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,hidden_dim_2 = 32, latent_dim=16):
        super(VAE, self).__init__()
        # Encoder：三層全連接網路
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim_2)  # 新增中間層
        self.fc21 = nn.Linear(hidden_dim_2, latent_dim)  # 輸出 mu
        self.fc22 = nn.Linear(hidden_dim_2, latent_dim)  # 輸出 logvar

        # Decoder：三層全連接網路
        self.fc3 = nn.Linear(latent_dim, hidden_dim_2)
        self.fc4 = nn.Linear(hidden_dim_2, hidden_dim)  # 新增中間層
        self.fc5 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc4(h3))
        return self.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# 根據 x_train_tensor 的維度設置 input_dim
input_dim = x_train_tensor.shape[1]

hidden_dim = 64     #encode 第一層隱藏層維度 and decode 第二層隱藏層維度
hidden_dim_2 = 32   #encode 第二層隱藏層維度 and decode 第一層隱藏層維度
latent_dim = 32     #encode 輸出層維度 decode 輸入層維度

model = VAE(input_dim=input_dim, hidden_dim=hidden_dim,hidden_dim_2 = hidden_dim_2, latent_dim=latent_dim).to(device)

# ---------------------------
# 3. 定義損失函數與優化器
# ---------------------------

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4. 訓練 VAE 模型
# ---------------------------

epochs = 100
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch in train_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# ---------------------------
# 5. 生成特定標籤的數據
# ---------------------------

def generate_specific_label_data(model, x_target):
    model.eval()
    with torch.no_grad():
        mu_target, logvar_target = model.encode(x_target.to(device))
        z_mean_specific = mu_target.mean(dim=0)
        z_std_specific = (0.5 * logvar_target).exp().mean(dim=0)
        latent_samples = z_mean_specific + torch.randn(496, latent_dim).to(device) * z_std_specific
        generated = model.decode(latent_samples)
    return generated.cpu().numpy()

# 調用 generate_specific_label_data() 並自行填入 x_target


mat_data_test = scipy.io.loadmat('input_data_models_1_dlp_0_F_10000.mat')
x_train_test = mat_data_test['Final_input_data']  # 載入變數

# 依據索引選擇特徵 
x_train_test = x_train_test[:, feature_indices]



x_train_scaled_test = (x_train_test - min_val) / (max_val - min_val)

# 將數據轉成 torch.Tensor
x_train_scaled_test = torch.tensor(x_train_scaled_test, dtype=torch.float32)
print("x_train_tensor shape:", x_train_scaled_test.shape)

generated = generate_specific_label_data(model,x_train_scaled_test)

generated_original = generated * (max_val - min_val) + min_val

test_predictions = model_1.predict(generated_original)

print("生成資料 shape:", generated_original.shape)

# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(train_predictions,'rx', label="origin data Predictions",markersize=5)
plt.plot(test_predictions,'bo', label="vae data Predictions",markersize=2)

# 圖例與標題
plt.xlabel("Sample Index")
plt.ylabel("Prediction Value")
plt.title("origin data vs vae data Predictions")
plt.legend()
plt.show()
