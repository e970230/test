import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist
import cv2
import os

# 街口：載入自定義圖片數據
def load_custom_images(image_dir, image_size):
    """
    從指定目錄中載入圖片，並調整為統一大小。
    
    Args:
        image_dir (str): 圖片存放的目錄路徑。
        image_size (tuple): 目標大小 (height, width)。
    
    Returns:
        numpy.ndarray: 調整大小後的圖片數據。
    """
    images = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
    images = np.array(images)
    # 正規化到 [0, 1]
    images = images.astype("float32") / 255.0
    # 展平成向量
    images = images.reshape((len(images), np.prod(image_size)))
    return images

# 下面是使用MNIST數據的基礎AE結構
def build_autoencoder(input_dim, encoding_dim):
    """
    建立自編碼器模型。
    
    Args:
        input_dim (int): 輸入向量的維度。
        encoding_dim (int): 潛在空間表示的維度。
    
    Returns:
        tuple: (autoencoder, encoder, decoder) 模型。
    """
    # 編碼器
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    # 解碼器
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # 自編碼器
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    # 獨立的解碼器
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1](encoded_input)
    decoder = Model(encoded_input, decoder_layer)

    return autoencoder, encoder, decoder

# 建立並訓練模型
def train_autoencoder(x_train, x_test, input_dim, encoding_dim):
    """
    訓練自編碼器。
    
    Args:
        x_train (numpy.ndarray): 訓練數據。
        x_test (numpy.ndarray): 測試數據。
        input_dim (int): 輸入數據的維度。
        encoding_dim (int): 潛在空間表示的維度。
    
    Returns:
        tuple: 訓練好的 autoencoder, encoder, decoder 模型。
    """
    autoencoder, encoder, decoder = build_autoencoder(input_dim, encoding_dim)

    # 編譯模型
    autoencoder.compile(optimizer='adam', loss='mse')

    # 訓練模型
    autoencoder.fit(x_train, x_train,
                    epochs=5,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    return autoencoder, encoder, decoder

# 測試與可視化結果
def visualize_results(x_test, decoded_imgs):
    """
    顯示原始圖片與重建圖片的對比。
    
    Args:
        x_test (numpy.ndarray): 原始測試圖片。
        decoded_imgs (numpy.ndarray): 重建的測試圖片。
    """
    n = min(10, len(x_test))  # 根據數據集大小調整要顯示的圖片數量
    #n = 2  # 顯示10張圖片
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

# 主程序流程
if __name__ == "__main__":
    # 使用MNIST數據進行初步測試
    (x_train, _), (x_test, _) = mnist.load_data()

    # 將數據標準化到 [0, 1] 並展平成向量
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # 自編碼器的輸入維度與潛在維度
    input_dim = x_train.shape[1]
    encoding_dim = 32

    # 訓練模型
    autoencoder, encoder, decoder = train_autoencoder(x_train, x_test, input_dim, encoding_dim)

    # 測試模型
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # 可視化結果
    visualize_results(x_test, decoded_imgs)

    # 載入自己的圖片
    custom_images_dir = r"C:\Users\admin\Desktop\DL"
    image_size = (28, 28)  # 假設圖片大小固定為28x28
    custom_images = load_custom_images(custom_images_dir, image_size)

    # 用訓練好的模型對自定義圖片進行測試
    encoded_custom = encoder.predict(custom_images)
    decoded_custom = decoder.predict(encoded_custom)

    # 可視化自定義圖片的重建結果
    visualize_results(custom_images, decoded_custom)
