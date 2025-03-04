from tensorflow import keras, data
from keras import layers
import numpy as np
from keras import backend as K
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from sklearn.preprocessing import StandardScaler

class VAE(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc21 = layers.Dense(latent_dim)  # Mean
        self.fc22 = layers.Dense(latent_dim)  # Log variance

        # Decoder
        self.fc3 = layers.Dense(hidden_dim, activation='relu')
        self.fc4 = layers.Dense(input_dim, activation='sigmoid')

    def encode(self, x):
        h1 = self.fc1(x)
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        std = tf.exp(0.5 * logvar)
        return mu + eps * std

    def decode(self, z):
        h3 = self.fc3(z)
        return self.fc4(h3)

    def call(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        self.add_loss(self.vae_loss(x, reconstructed, mu, logvar))  # Adding loss inside the model
        return reconstructed

    def vae_loss(self, x, recon_x, mu, logvar):
        bce = tf.keras.losses.binary_crossentropy(x, recon_x)
        bce = tf.reduce_sum(bce, axis=-1)
        kld = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)
        return tf.reduce_mean(bce + kld)

#x_train = scipy.io.loadmat(f'input_data_models_1_F_10000.mat')['train_input_data']

# Load Data

x_train = scipy.io.loadmat(f'input_data_models_1_dlp_40_F_10000.mat')['Final_input_data']
# Load model and best feature
best_feature_models_1_F_10000 = pd.read_csv("best_solution_models_1_F_10000.csv")
model_1 = pickle.load(open('models_1_F_10000.pkl', "rb"))
feature_indices = best_feature_models_1_F_10000.iloc[9:-2, 1].astype(int).values

filtered_test_data = x_train[:, feature_indices]
train_predictions = model_1.predict(filtered_test_data)

input_dim = np.shape(x_train)[1] #原始資料的特徵數量

std=StandardScaler()

x_train=std.fit_transform(x_train) #對數據進行正規化


batch_size = 512        #每次訓練時處理的數據樣本數量。
hidden_dim = 400        #隱藏層大小
latent_dim = 2          #編碼大小
sample_number = 500     #生成樣本數
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(len(x_train)).batch(batch_size)

# Initialize VAE and Train
vae = VAE(input_dim, hidden_dim, latent_dim)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(train_dataset, epochs=10)



mu, logvar=vae.encode(x_train)

global_mu=np.mean(mu.numpy(), axis=0)
global_lovar=np.sqrt(np.mean(np.exp(logvar.numpy()),axis=0))


# Generate new data
latent_samples=np.random.normal(global_mu,global_lovar,size=(sample_number,latent_dim)).astype(np.float32)



generated_data = vae.decode(latent_samples)

generated_data=std.inverse_transform(generated_data)

print("Generated data shape:", generated_data.shape)




# 確保 feature_indices 不超過 generated_data 的範圍
max_feature_index = generated_data.shape[1] - 1
feature_indices = feature_indices[feature_indices <= max_feature_index]

# 轉換為 NumPy 數組
feature_indices = np.array(feature_indices, dtype=np.int32)

# 確保 generated_data 是 NumPy 陣列
if isinstance(generated_data, tf.Tensor):
    generated_data = generated_data.numpy()

# 確保 feature_indices 是 list
feature_indices = feature_indices.tolist()

# 進行索引

generated_test_data = generated_data[:, feature_indices]


test_predictions = model_1.predict(generated_test_data)

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