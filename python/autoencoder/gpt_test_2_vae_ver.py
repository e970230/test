import numpy as np
import tensorflow as tf
import scipy.io
import pickle
import pandas as pd
import matplotlib.pyplot as plt

class MeanSquaredError(tf.keras.Loss):
    def call(self, y_true, y_pred):
        return tf.keras.ops.mean(tf.keras.ops.square(y_pred - y_true), axis=-1)

    
class KLLoss(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
class vae_loss_layer(tf.keras.layers.Layer):
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

#x_train = scipy.io.loadmat(f'input_data_models_1_F_10000.mat')['train_input_data']
x_train = scipy.io.loadmat(f'input_data_models_1_dlp_0_F_10000.mat')['Final_input_data']
#x_test = scipy.io.loadmat(f'test_input_models_1_F_10000.mat')['test_input_data']

best_feature_models_1_F_10000 = pd.read_csv("best_solution_models_1_F_10000.csv")
model_1 = pickle.load(open('models_1_F_10000.pkl', "rb"))

# 取出第 10 列至倒數第3列的特徵索引 
feature_indices = best_feature_models_1_F_10000.iloc[9:-2, 1].astype(int).values

# 篩選測試數據，只保留選取的特徵
filtered_test_data = x_train[:, feature_indices]


predictions = model_1.predict(filtered_test_data)

plt.figure(figsize = (10,6))
plt.plot(predictions)
plt.show()

#x_test = np.random.rand(1000, original_dim).astype('float32')

# 参数
original_dim = np.shape(x_train)[1]  # 输入的维度
latent_dim = 2     # 潜在空间的维度
intermediate_dim = 50

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
vae = tf.keras.models.Model(inputs, outputs)
vae.summary()




#reconstruction_loss = tf.keras.losses.MeanSquaredError()(inputs, outputs)
#kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
# 损失函数
reconstruction_loss = MeanSquaredError()(inputs, outputs)
kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae_loss = reconstruction_loss + kl_loss
vae.add_loss(vae_loss)



# 添加到模型中


vae.compile(optimizer='adam',loss = 'mse')


# 训练模型
vae.fit(x_train, x_train, epochs=10, batch_size=32)

# 测试：生成新数据
decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
decoder = tf.keras.Model(decoder_input, _x_decoded_mean)
decoder.summary()


latent_samples = np.random.normal(size=(20, latent_dim))
generated_data = decoder.predict(latent_samples)

print("Generated data shape:", generated_data.shape)

# 篩選測試數據，只保留選取的特徵
generated_test_data = generated_data[:, feature_indices]


test_predictions = model_1.predict(generated_test_data)

plt.figure(figsize = (10,6))
plt.plot(test_predictions)
plt.show()