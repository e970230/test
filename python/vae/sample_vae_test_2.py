import numpy as np
import tensorflow as tf

def sampling(args):
    """取樣函數，用於從學到的分佈中生成新數據"""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# 設定模型參數
original_dim = 100  # 假設每個數據樣本是 100 維向量
latent_dim = 10     # 潛在空間的維度
intermediate_dim = 50

# 建立模型
# 編碼器
inputs = tf.keras.Input(shape=(original_dim,))
x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = tf.keras.layers.Dense(latent_dim)(x)
z_log_var = tf.keras.layers.Dense(latent_dim)(x)
z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

# 解碼器
decoder_h = tf.keras.layers.Dense(intermediate_dim, activation='relu')
decoder_mean = tf.keras.layers.Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# VAE 模型
vae = tf.keras.Model(inputs, x_decoded_mean)

# 使用實際數據進行 MSE 計算
original_data = np.random.rand(1000, original_dim).astype(np.float32)
predicted_data = vae.predict(original_data)




# 將損失加入模型

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss=tf.keras.losses.MeanSquaredError()(original_data, predicted_data))

# 訓練 VAE
vae.fit(original_data, epochs=10, batch_size=32)

# 創造新數據樣本
# 我們可以從標準正態分佈中抽樣 latent variables，然後通過解碼器生成新數據
decoder_input = tf.keras.Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = tf.keras.Model(decoder_input, _x_decoded_mean)

# 隨機抽樣新的潛在變量，生成更多數據
latent_samples = np.random.normal(size=(10, latent_dim))
new_data_samples = generator.predict(latent_samples)

print("Generated data shape:", new_data_samples.shape)
