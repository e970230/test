import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import random_normal, sum, square, exp

def sampling(args):
    """从 z_mean 和 z_log_var 中采样潜在变量"""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = random_normal(shape=(batch, dim))
    return z_mean + exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim):
    """
    建立变分自编码器模型。
    input_dim: 输入数据的维度
    latent_dim: 潜在空间的维度
    """
    inputs = Input(shape=(input_dim,))
    outputs = Dense(input_dim, activation='sigmoid')(inputs)
    vae = Model(inputs, outputs)

    mse = MeanSquaredError()
    reconstruction_loss = mse(inputs, outputs)
    vae.add_loss(reconstruction_loss)
    vae.compile(optimizer='adam')

    return vae

if __name__ == "__main__":
    
    from tensorflow.keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()

    # 数据预处理
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    input_dim = x_train.shape[1]
    latent_dim = 2

    # 创建 VAE 模型并训练
    vae, decoder = build_vae(input_dim)
    vae.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

    # 生成新样本
    z_samples = np.random.normal(size=(10, latent_dim))
    generated_images = decoder.predict(z_samples)

    # 可视化生成的图片
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 4))
    for i in range(10):
        ax = plt.subplot(1, 10, i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.axis("off")
    plt.show()
