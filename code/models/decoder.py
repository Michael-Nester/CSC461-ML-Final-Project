import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

# Load saved data
latent_vectors = np.load('../csc461/brn/latent_vectors.npy')
image_data = np.load('../csc461/brn/image_data.npy')

# Enhanced decoder architecture
latent_dim = 64
latent_input = tf.keras.Input(shape=(latent_dim,))

# Improved decoder layers with more filters and BatchNormalization
x = Dense(28 * 28 * 128, activation='relu')(latent_input)
x = BatchNormalization()(x)
x = Reshape((28, 28, 128))(x)

x = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
x = BatchNormalization()(x)

decoded_output = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

# Create and compile decoder model
decoder = Model(latent_input, decoded_output, name="decoder")
decoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss='mse'
)

# Train the decoder
print("Training decoder...")
decoder.fit(
    latent_vectors,
    image_data,
    epochs=100,
    batch_size=16,
    shuffle=True,
    validation_split=0.1
)

# Save the trained decoder
decoder.save('../csc461/brn/decoder/decoder_model.h5')
