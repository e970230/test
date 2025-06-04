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
mat_data = scipy.io.loadmat('input_data_models_1_F_10000.mat')
x_train = mat_data['train_input_data']

best_feature_models_1_F_10000 = pd.read_csv("best_solution_models_1_F_10000.csv")
model_1 = pickle.load(open('models_1_F_10000.pkl', "rb"))
feature_indices = best_feature_models_1_F_10000.iloc[9:-2, 1].astype(int).values
x_train = x_train[:, feature_indices]

mat_data_output = scipy.io.loadmat('output_data_models_1_F_10000.mat')
y_train = mat_data_output['train_output_data'] 

train_predictions = model_1.predict(x_train)

min_val = np.min(x_train)
max_val = np.max(x_train)
x_train_scaled = (x_train - min_val) / (max_val - min_val)

x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)


# 建立 Dataset 與 DataLoader
dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------------------
# 2. 定義 CVAE 模型（實數條件）
# ---------------------------
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=64, hidden_dim_2=32, latent_dim=16):
        super(CVAE, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim_2)
        self.fc_mu = nn.Linear(hidden_dim_2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_2, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim_2)
        self.decoder_fc2 = nn.Linear(hidden_dim_2, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h1 = self.relu(self.encoder_fc1(xy))
        h2 = self.relu(self.encoder_fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h3 = self.relu(self.decoder_fc1(zy))
        h4 = self.relu(self.decoder_fc2(h3))
        return self.sigmoid(self.decoder_out(h4))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar

input_dim = x_train_tensor.shape[1]
condition_dim = y_train_tensor.shape[1]
model = CVAE(input_dim=input_dim, condition_dim=condition_dim).to(device)

# ---------------------------
# 3. 損失函數與優化器
# ---------------------------
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4. 訓練模型
# ---------------------------
epochs = 1000
model.train()
for epoch in range(epochs):
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, y)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# ---------------------------
# 5. 生成資料（使用實數條件）
# ---------------------------
model.eval()
with torch.no_grad():
    m = 496
    latent_vectors = torch.randn(m, model.fc_mu.out_features).to(device)

    # 你可以在這裡指定生成時的實數條件值
    y_generate = np.full((m, condition_dim), 22, dtype=np.float32)  # 例如條件值為 0.75
    y_generate_tensor = torch.tensor(y_generate).to(device)

    generated = model.decode(latent_vectors, y_generate_tensor).cpu().numpy()

generated_original = generated * (max_val - min_val) + min_val
test_predictions = model_1.predict(generated_original)

print("生成資料 shape:", generated_original.shape)

plt.figure(figsize=(10, 6))
plt.plot(train_predictions, 'rx', label="origin data Predictions", markersize=5)
plt.plot(test_predictions, 'bo', label="cvae data Predictions", markersize=2)
plt.xlabel("Sample Index")
plt.ylabel("Prediction Value")
plt.title("origin data vs cvae data Predictions")
plt.legend()
plt.show()
