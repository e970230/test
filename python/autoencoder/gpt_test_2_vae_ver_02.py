import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer

class ReconstructionLoss(Layer):
    def call(self, inputs):
        x_true, x_pred = inputs
        return tf.keras.losses.MeanSquaredError()(x_true, x_pred)
    
class KLLoss(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
class vae_loss_layer(Layer):
    def call(self, inputs):
        reconstruction_loss, kl_loss = inputs
        return reconstruction_loss + kl_loss

def sampling(args):
    """取样函数，用于从学到的分布中生成潜在变量"""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# 参数
original_dim = 28 * 28  # 输入的维度
latent_dim = 10         # 潜在空间的维度
intermediate_dim = 128  # 隐藏层维度

# 编码器
inputs = tf.keras.layers.Input(shape=(original_dim,))
h = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(h)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(h)
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码器
decoder_h = tf.keras.layers.Dense(intermediate_dim, activation='relu')
decoder_mean = tf.keras.layers.Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_mean(h_decoded)

# VAE 模型
vae = tf.keras.Model(inputs, outputs)

# 使用均方误差 (MSE) 作为损失函数
vae.compile(optimizer='adam', loss='mse')

# 加载 MNIST 数据
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.  # 归一化到 [0, 1]
x_test = x_test.astype('float32') / 255.

# 将图像展平到一维
x_train = x_train.reshape((len(x_train), original_dim))
x_test = x_test.reshape((len(x_test), original_dim))

# 训练模型
vae.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

# 重建测试图片
decoded_imgs = vae.predict(x_test[:10])

# 可视化原始图片和重建图片
n = 10  # 显示前10张
plt.figure(figsize=(20, 4))
for i in range(n):
    # 原始图片
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.axis('off')

    # 重建图片
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.show()
