import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist

# 加載MNIST數據集
(x_train, _), (x_test, _) = mnist.load_data()

# 將數據標準化到[0, 1]並展平成向量
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 定義編碼器和解碼器的層數和神經元數
input_dim = x_train.shape[1]
encoding_dim = 32  # 壓縮到32個維度

# 編碼器模型
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)

# 解碼器模型
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# 自編碼器模型
autoencoder = Model(input_img, decoded)

# 編碼器單獨模型
encoder = Model(input_img, encoded)

# 構建解碼器模型
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1](encoded_input)
decoder = Model(encoded_input, decoder_layer)

# 編譯模型
autoencoder.compile(optimizer='adam', loss='mse')

# 訓練自編碼器
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# 壓縮測試集並重建
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# 可視化原始圖片與重建圖片
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 原始圖片
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # 重建圖片
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")
plt.show()
