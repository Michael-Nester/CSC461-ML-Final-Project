import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
IMG_SIZE = 128
LATENT_DIM = 8
BATCH_SIZE = 32
NUM_EPOCHS = 50

# Load and preprocess data
image_folder = '../csc461/brn/trainData/'
csv_path = '../csc461/brn/train_labels.csv'

# Load and sort the CSV file
labels_df = pd.read_csv(csv_path)
labels_df['C'] = labels_df['filename'].str.extract(r'C(\d+)')[0].astype(int)
labels_df['S'] = labels_df['filename'].str.extract(r'S(\d+)')[0].astype(int)
labels_df['I'] = labels_df['filename'].str.extract(r'I(\d+)')[0].astype(int)
labels_df = labels_df.sort_values(['C', 'S', 'I'])

image_names = labels_df['filename']
labels = labels_df['label']

# Load and preprocess images
image_data = []
image_labels = []

for i, image_name in enumerate(image_names):
    image_path = os.path.join(image_folder, image_name)
    if os.path.exists(image_path):
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = img_array.astype("float32") / 255.0
        image_data.append(img_array)
        image_labels.append(labels[i])

image_data = np.array(image_data)
image_labels = np.array(image_labels)


# Create the encoder
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# Downsample from 128x128 -> 64x64 -> 32x32 -> 16x16
x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(input_img)
x = BatchNormalization()(x)
x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
latent_vector = Dense(LATENT_DIM)(x)

# Create the decoder
decoder_input = Input(shape=(LATENT_DIM,))
x = Dense(8 * 8 * 256)(decoder_input)
x = Reshape((8, 8, 256))(x)
# Upsample from 16x16 -> 32x32 -> 64x64 -> 128x128
x = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x)


# Create the models
encoder = Model(input_img, latent_vector, name='encoder')
decoder = Model(decoder_input, x, name='decoder')
autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

# Compile the model
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

# Create checkpoint directory
checkpoint_dir = '../csc461/brn/autoencoder/checkpoints8'
os.makedirs(checkpoint_dir, exist_ok=True)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(checkpoint_dir, 'best_model_f8.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
history = autoencoder.fit(
    image_data,
    image_data,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,
    callbacks=callbacks
)

# Save the models
autoencoder.save('../csc461/brn/autoencoder_model_f8')
encoder.save('../csc461/brn/encoder_model_f8')
decoder.save('../csc461/brn/decoder_model_f8')


# Save latent vectors
latent_vectors = encoder.predict(image_data)
np.save('../csc461/brn/latent_vectors_f8.npy', latent_vectors)


# Save the history
history_dict = history.history
np.save('../home/csc461/brn/training_history_f8.npy', history_dict)

# Save mapping
vector_mapping = pd.DataFrame({
    'filename': image_names,
    'label': labels,
})
vector_mapping.to_csv('../csc461/brn/vector_mapping_f8.csv', index=False)

print(f"Training complete. Models and reconstructions saved.")
print(f"Latent vectors shape: {latent_vectors.shape}")


print(f"Generating and saving comparison between original image and recostructions. This may take a while.")
# Generate and save reconstructions
reconstructed_images = autoencoder.predict(image_data)

# Create output directory for reconstructed images
output_dir = '../csc461/brn/reconstructed_images8'
os.makedirs(output_dir, exist_ok=True)
"""
# Save reconstructed images in order
for i, (img, filename) in enumerate(zip(reconstructed_images, image_names)):
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_data[i])
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('Reconstructed')
    plt.axis('off')
    
    # Save with original filename (but as PNG)
    base_name = os.path.splitext(filename)[0]
    plt.savefig(os.path.join(output_dir, f'{base_name}_comparison.png'))
    plt.close()

"""
