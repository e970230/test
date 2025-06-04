import numpy as np
import matplotlib.pyplot as plt
# Import some useful packages
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider, FloatSlider

import tensorflow as tf
import tensorflow.keras.backend as K

# Layers for FNN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Lambda, concatenate
from tensorflow.keras.layers import Dense

# Optimizers for training
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import metrics

# Losses for training
from tensorflow.keras import losses

# For data preprocessing
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer

class ReconstructionLoss(Layer):
    def call(self, inputs):
        x, x_hat = inputs
        return tf.keras.losses.MeanSquaredError(x, x_hat)*784
    
class KLLoss(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss
    


def reconstructed_img(idx):
    # Prepare image to be reconstructed
    X = X_train[idx:idx+1]
    y = y_train0[idx]
    X_hat = autoencoder.predict(X)

    # Reshape for plotting  
    X = X.reshape(28, 28)
    X_hat = X_hat.reshape(28, 28)
    rec_error = np.abs(X-X_hat)

    # Prepare a canvas for plotting
    plt.figure(figsize=(12, 9))
    ax1 = plt.subplot2grid((2,3),(0,0))
    ax2 = plt.subplot2grid((2,3),(0,1))
    ax3 = plt.subplot2grid((2,3),(0,2))
    ax4 = plt.subplot2grid((2,3),(1,0), colspan=3)

    # Plot raw image
    ax1.imshow(X, 'Greys')
    ax1.set_title(f'Number: {y}')

    # Plot reconstructed image
    ax2.imshow(X_hat, 'Greys')
    ax2.set_title('Reconstructed Image')

    # Plot reconstruction error
    ax3.imshow(rec_error, 'Greys')
    ax3.set_title('Reconstructed Error')
    ax4.hist(rec_error.flatten(), bins=20, rwidth=0.5)
    ax4.set_title('Histogram of Reconstructed Error')


# Load dataset
(X_train, y_train0), (X_test, y_test0) = datasets.mnist.load_data()

# Reshape size
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Normalize the range of featurs
X_train = X_train / X_train.max()
X_test = X_test / X_test.max()

# One-hot encoding
y_train = to_categorical(y_train0, 10)
y_test = to_categorical(y_test0, 10)

x = Input(shape=(784,))

enc_1 = Dense(100, activation='sigmoid')
enc_2 = Dense(2, activation='sigmoid')

# Define latent repre. of x
z = enc_2(enc_1(x))

dec_2 = Dense(100, activation='sigmoid')
dec_1 = Dense(784, activation='sigmoid')

x_hat = dec_1(dec_2(z))
autoencoder = Model(x, x_hat)
autoencoder.summary()

autoencoder.compile(loss='mse',
                    optimizer=Adam(5e-3),
                    metrics=['mae']
                    )

# autoencoder.load_weights('autoencoder_mse_mae_weights.h5')
autoencoder.fit(X_train, X_train, 
                batch_size=512, 
                epochs=5, 
                validation_data=(X_test, X_test)
                )

# autoencoder.save_weights('autoencoder_mse_mae_weights.h5')

# Plot
interact(reconstructed_img, idx=IntSlider(min=0, max=X_train.shape[0], step=1, value=3));

Encoder = Model(x, z)
Encoder.summary()

z_input = Input(shape=(2,))
Decoder = Model(z_input, dec_1(dec_2(z_input)))

Decoder.summary()

idx = np.random.randint(X_train.shape[0])
print(f"第 {idx} 圖的 latent 表示為 {Encoder.predict(X_train[idx: idx+1]).squeeze()}")

indices = np.random.randint(X_train.shape[0], size=3000)
latents = Encoder.predict(X_train[indices])
plt.scatter(latents[:, 0], latents[:, 1], c=y_train0[indices], cmap="tab10")
plt.colorbar()
plt.show()


n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(0.05, 0.95, n)
grid_y = np.linspace(0.05, 0.95, n)
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = Decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[(n-i-1) * digit_size: (n - i) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
XX, YY = np.meshgrid(grid_x, grid_y)
plt.scatter(XX, YY)

plt.subplot(1, 2, 2)
plt.imshow(figure, cmap='Greys')
plt.axis('off')
plt.show()

#------------------------VAE-------------------------------------------

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def normalized(x):
    x -= x.min()
    x /= x.max()
    return x

enc_1 = Dense(100, activation='sigmoid')
# enc_2 = Dense(2, activation='sigmoid')

enc_mean = Dense(2)
enc_log_var = Dense(2)

dec_2 = Dense(100, activation='sigmoid')
dec_1 = Dense(784, activation='sigmoid')

x = Input(shape=(784,))
enc_x = enc_1(x)

z_mean = enc_mean(enc_x)
z_log_var = enc_log_var(enc_x)

# Sampling function wrapped as a Keras layer
z = Lambda(sampling, output_shape=(2,))([z_mean, z_log_var])
# Define Decoder part of VAE
z_input = Input(shape=(2,))
x_hat = dec_1(dec_2(z_input))
x_hat = dec_1(dec_2(z))

VAE = Model(x, x_hat)
VAE.summary()

# reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
reconstruction_loss = 784 * losses.MeanSquaredError(x, x_hat)

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)

VAE.add_loss(vae_loss)

VAE.compile(optimizer=Adam(),loss='mse')

# VAE.load_weights('VAE_handwriting_model_weights.h5')
VAE.fit(X_train, X_train, 
        batch_size=512, 
        epochs=5)

VAE.save_weights('VAE_handwriting_model.weights.h5')

VAE_Encoder = Model(x, z_mean)

VAE_Encoder.summary()

VAE_Decoder = Model(z_input, dec_1(dec_2(z_input)))

VAE_Decoder.summary()

idx = np.random.randint(X_train.shape[0])
print(f"第 {idx} 圖的 latent 表示為 {VAE_Encoder.predict(X_train[idx: idx+1]).squeeze()}")

VAE_latents = VAE_Encoder.predict(X_train[indices])
plt.figure(figsize=(10, 8))
plt.scatter(VAE_latents[:, 0], VAE_latents[:, 1], c=y_train0[indices], cmap='tab20')
plt.colorbar()
plt.show()

grid_x_vae = np.linspace(-4+0.05, 4-0.05, n)
grid_y_vae = np.linspace(-4+0.05, 4-0.05, n)
VAE_figure = np.zeros((digit_size * n, digit_size * n))
for i, yi in enumerate(grid_x_vae):
    for j, xi in enumerate(grid_y_vae):
        z_sample = np.array([[xi, yi]])
        x_decoded = VAE_Decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        VAE_figure[(n-i-1) * digit_size: (n - i) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = normalized(digit)
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
XXX, YYY = np.meshgrid(grid_x_vae, grid_y_vae)
plt.scatter(XXX, YYY)

plt.subplot(1, 2, 2)
plt.imshow(figure, cmap='Greys')
plt.axis('off')
plt.show()

#AE VS VAE

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.scatter(VAE_latents[:, 0], VAE_latents[:, 1], c=y_train0[indices], cmap='tab10')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(latents[:, 0], latents[:, 1], c=y_train0[indices], cmap="tab10")
plt.colorbar()
plt.show()

idx_1, idx_2 = np.random.randint(X_train.shape[0], size=2)
def inBetween(t):
    data_0 = X_train[idx_1].reshape(28, 28)
    data_1 = X_train[idx_2].reshape(28, 28)
    data_t = (1-t)*data_0 + t*data_1
    
    mu_0 = VAE_Encoder.predict(X_train[idx_1:idx_1+1]).squeeze()
    mu_1 = VAE_Encoder.predict(X_train[idx_2:idx_2+1]).squeeze()
    mu_t = (1-t)*mu_0 + t*mu_1

    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(2, 1, 2)
    ax1.scatter(mu_0[0], mu_0[1])
    ax1.scatter(mu_1[0], mu_1[1])
    ax1.scatter(mu_t[0], mu_t[1])

    ax2 = plt.subplot(2, 3, 1)
    ax2.imshow(data_0, cmap='Greys')
    ax2.set_title('t=0')

    ax3 = plt.subplot(2, 3, 2)
    ax3.imshow(data_t, cmap='Greys')
    ax3.set_title(f't={t}')

    ax4 = plt.subplot(2, 3, 3)
    ax4.imshow(data_1, cmap='Greys')
    ax4.set_title('t=1')
interact(inBetween, t=FloatSlider(value=0, 
                                  min=0, 
                                  max=1.0,
                                  step=0.02,))