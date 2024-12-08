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

# ... existing imports ...
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ... existing code until training section ...

# Create checkpoint directory if it doesn't exist
checkpoint_dir = '../csc461/brn/decoder/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Number of epochs to wait before stopping if no improvement
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(checkpoint_dir, 'best_model.h5'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the decoder with callbacks
print("Training decoder...")
history = decoder.fit(
    latent_vectors,
    image_data,
    epochs=100,
    batch_size=16,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint]
)

# Find epoch with lowest validation loss
best_epoch = np.argmin(history.history['val_loss']) + 1
print(f"\nBest epoch was {best_epoch} with validation loss: {min(history.history['val_loss']):.4f}")

# Model is already restored to best weights due to restore_best_weights=True
decoder.save('../csc461/brn/decoder/decoder_model.h5')

