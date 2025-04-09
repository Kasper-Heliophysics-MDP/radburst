import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
import numpy as np

def build_denoising_autoencoder(input_shape=(190, 190, 1), latent_dim=128, device='/CPU:0'):
    input_img = Input(shape=input_shape)

    # --- Encoder with vertically-oriented kernels ---
    x = Conv2D(32, (7, 3), padding='same', kernel_initializer='he_normal')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (5, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]

    x = Flatten()(x)
    latent = Dense(latent_dim, activation='relu')(x)

    # --- Decoder ---
    x = Dense(np.prod(shape_before_flatten), activation='relu')(latent)
    x = Reshape(shape_before_flatten)(x)

    x = Conv2D(64, (5, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)  # (48 → 96)

    x = Conv2D(32, (7, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)  # (96 → 192)

    # NEW: Reduce from 192x192 → 190x190 using valid padding
    decoded = Conv2D(1, (3, 3), activation='linear', padding='valid')(x)

    with tf.device(device):
        autoencoder = Model(inputs=input_img, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder