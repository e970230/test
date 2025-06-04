import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
 
from keras.datasets import mnist
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.losses import binary_crossentropy
 
# network parameters
rec_dim=784
input_shape = (rec_dim,)
int_dim = 512
lat_dim = 2
 
# Load the MNIST data
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
 
# normalize values of image pixels between 0 and 1f
x_tr = x_tr.astype('float32') / 255.
x_te = x_te.astype('float32') / 255.
 
# 28x28 2D matrix --> 784x1 1D vector
x_tr = x_tr.reshape((len(x_tr), np.prod(x_tr.shape[1:])))
x_te = x_te.reshape((len(x_te), np.prod(x_te.shape[1:])))
 
print(x_tr.shape, x_te.shape)

#=======================
# Encoder
#=======================
# Z sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    # Reparameterization Trick
    # draw random sample ε from Gussian(=normal) distribution
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
 
# Input shape
inputs = Input(shape=input_shape)
enc_x  = Dense(int_dim, activation='relu')(inputs)
 
z_mean    = Dense(lat_dim)(enc_x)
z_log_var = Dense(lat_dim)(enc_x)
 
# sampling z
z_sampling = Lambda(sampling, (lat_dim,))([z_mean, z_log_var])
 
# encoder model has multi-output so a list is used
encoder = Model(inputs,[z_mean,z_log_var,z_sampling])
encoder.summary()
 
#=======================
# Decoder
#=======================
# Input of decoder is z
input_z = Input(shape=(lat_dim,))
dec_h   = Dense(int_dim, activation='relu')(input_z)
outputs = Dense(rec_dim, activation='sigmoid')(dec_h)
 
# z is the input and the reconstructed image is the output
decoder = Model(input_z, outputs)
decoder.summary()

#=======================
# VAE model
#=======================
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs)
 
#--------------------------------------------------
# VAE_loss = ELBO
#--------------------------------------------------
# (1)Reconstruct loss (Marginal_likelihood) : Cross-entropy 
rec_loss = tf.keras.losses.binary_crossentropy(inputs,outputs)
rec_loss *= rec_dim
# (2) KL divergence(Latent_loss)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = -0.5*K.sum(kl_loss, 1)
# (3) ELBO
vae_loss = K.mean(rec_loss + kl_loss)
#--------------------------------------------------
 
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
 
history = vae.fit(x_tr, x_tr, shuffle=True, 
                  epochs=30, batch_size=64, 
                  validation_data=(x_te, x_te))