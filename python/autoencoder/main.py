from tensorflow import keras ,data
from keras import layers
import numpy as np
from keras import backend as K
import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc21 = layers.Dense(latent_dim)  # 均值
        self.fc22 = layers.Dense(latent_dim)  # 標準差（log variance）

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
        self.add_loss(self.vae_loss(x, reconstructed, mu, logvar))  # 直接在模型內部添加 loss
        return reconstructed

    def vae_loss(self, x, recon_x, mu, logvar):
        # 計算 BCE（重建誤差）
        bce = tf.keras.losses.binary_crossentropy(x, recon_x)
        bce = tf.reduce_sum(bce, axis=-1)

        # 計算 KLD（KL 散度）
        kld = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)

        return tf.reduce_mean(bce + kld)


# 1. 加載 MNIST 數據
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)  # 轉換為 784 維

# 2. 創建數據集
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)

# 3. 初始化 VAE 並訓練
vae = VAE()
vae.compile(optimizer=tf.keras.optimizers.Adam())

# 4. 訓練模型（不需要顯式傳遞 Loss，因為 `self.add_loss()` 已經定義了）
vae.fit(train_dataset, epochs=10)