import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import numpy as np

def build_denoising_autoencoder(input_shape=(128, 128, 1), latent_dim=128):
    # Define the input layer
    input_img = Input(shape=input_shape)
    
    # --- Encoder ---
    # First convolutional layer followed by pooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Second convolutional layer followed by pooling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Save the shape for later use in the decoder
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    
    # Flatten and add the latent (dense) layer
    x = Flatten()(x)
    latent = Dense(latent_dim, activation='relu')(x)
    
    # --- Decoder ---
    # Fully connected layer to initiate reconstruction
    x = Dense(np.prod(shape_before_flatten), activation='relu')(latent)
    x = Reshape(shape_before_flatten)(x)
    
    # First convolutional layer in the decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # First upsampling layer to increase resolution
    x = UpSampling2D((2, 2))(x)
    
    # Second convolutional layer in the decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # Second upsampling layer to restore original resolution
    x = UpSampling2D((2, 2))(x)
    
    # Output layer: final convolution to produce the denoised image
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Build and compile the autoencoder model
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

if __name__ == '__main__':
    model = build_denoising_autoencoder()
    model.summary()
