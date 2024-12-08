
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array



image_folder = '../csc461/brn/filtered_iris'
csv_path = '../csc461/brn/filtered_labels.csv'

# Load the CSV file into a DataFrame
labels_df = pd.read_csv(csv_path)

# Display the first few rows to verify
print(labels_df.head())


image_names = labels_df['filename']
labels = labels_df['label']



# Load and preprocess images
image_data = []
image_labels = []

for i, image_name in enumerate(image_names):
    # Handle potential extension mismatches (e.g., .tiff -> .jpg)
    image_name = image_name.replace('.tiff', '.jpg')
    image_path = os.path.join(image_folder, image_name)

    # Check if the image exists
    if os.path.exists(image_path):
        try:
            # Load the image, resize to 28x28 (or your desired size), and convert to array
            img = load_img(image_path, target_size=(28, 28))  # Resize to 28x28
            img_array = img_to_array(img)  # Convert to array
            img_array = img_array.astype("float32") / 255.0  # Normalize to [0, 1]

            # Append to lists
            image_data.append(img_array)
            image_labels.append(labels[i])  # Append corresponding label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    else:
        print(f"File not found: {image_path}")

# Convert to NumPy arrays
image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(f"Number of images loaded: {len(image_data)}")
print(f"Image data shape: {image_data.shape}")

# Create encoder-only model
latent_dim = 64
input_img = Input(shape=(28, 28, 3), name="input_image")
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
latent_vector = Dense(latent_dim, activation='relu', name="latent_vector")(x)

# Create the encoder model
encoder = Model(inputs=input_img, outputs=latent_vector, name="encoder")

# If you have a trained autoencoder, load its weights
# autoencoder = tf.keras.models.load_model('path_to_saved_autoencoder')
# encoder.set_weights(autoencoder.layers[:5].get_weights())  # Adjust the number of layers as needed

# Generate latent vectors for all images
latent_vectors = encoder.predict(image_data)

# Save the latent vectors
np.save('../csc461/brn/latent_vectors2.npy', latent_vectors)

# Save original image data for comparison
np.save('../csc461/brn/image_data2.npy', image_data)

# Save labels
np.save('../csc461/brn/image_labels2.npy', image_labels)

# Save the encoder model
encoder.save('../csc461/brn/encoder_model2.h5')

# Optional: Save the mapping between filenames and their latent vectors
vector_mapping = pd.DataFrame({
    'filename': image_names,
    'label': labels,
})
vector_mapping.to_csv('../csc461/brn/vector_mapping2.csv', index=False)

print(f"Latent vectors shape: {latent_vectors.shape}")

