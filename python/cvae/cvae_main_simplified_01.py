import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

print("torch.cuda.is_available():", torch.cuda.is_available())
print("可用 GPU 數量:", torch.cuda.device_count())
print("目前使用的 GPU 名稱:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "無")

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備：{device}")

# ----------------------------
# 1. 載入 MNIST 資料並正規化至 [0, 1]
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor()  # 自動將像素縮放到 [0, 1]
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)

# ----------------------------
# 2. 定義簡化版 CVAE 模型（Encoder 不含條件）
# ----------------------------
class SimplifiedCVAE(nn.Module):
    def __init__(self, input_dim=784, num_classes=10, latent_dim=2, hidden_dim=400, hidden_dim_2=200):
        super(SimplifiedCVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        return self.decoder(zy)

    def forward(self, x, y):
        mu, logvar = self.encode(x)  # 不包含條件 y
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# ----------------------------
# 3. 定義損失函數
# ----------------------------
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ----------------------------
# 4. 訓練模型
# ----------------------------
model = SimplifiedCVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 3000
model.train()
for epoch in range(epochs):
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.view(-1, 28 * 28).to(device)
        y_onehot = torch.nn.functional.one_hot(y_batch, num_classes=10).float().to(device)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x_batch, y_onehot)
        loss = loss_function(x_recon, x_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

# ----------------------------
# 5. 生成數字 5 的手寫圖片
# ----------------------------
model.eval()
with torch.no_grad():
    z = torch.randn(16, 2).to(device)
    y_label = torch.full((16,), 7, dtype=torch.long)
    y_onehot = torch.nn.functional.one_hot(y_label, num_classes=10).float().to(device)
    generated = model.decode(z, y_onehot).cpu().view(-1, 28, 28)

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, img in zip(axes.flatten(), generated):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
plt.suptitle("Generated Digits for Label 7 (Simplified CVAE)")
plt.show()
