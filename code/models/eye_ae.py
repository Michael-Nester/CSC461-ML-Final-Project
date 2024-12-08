import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
tf.debugging.set_log_device_placement(True)

# Constants
IMG_SIZE = 128
LATENT_DIM = 64
COLOR_MAPPING = {
    'blue': 0,
    'brown': 1,
    'green': 2,
    'hazel': 3,
    'gray': 4
}

def load_ubiris_data(image_folder, csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Initialize lists for images and labels
    images = []
    labels = []
    # Load and preprocess each image
    for idx, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            # Resize and normalize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(COLOR_MAPPING[row['label']])

    return np.array(images), np.array(labels)

def get_class_averages(encoder, images, labels):
    # Get encoded vectors
    encoded_vectors = encoder.predict(images)

    # Initialize dictionary to store vectors by class
    class_vectors = {}

    # Group vectors by class
    for color, idx in COLOR_MAPPING.items():
        class_mask = labels == idx
        class_vectors[color] = np.mean(encoded_vectors[class_mask], axis=0)

    return class_vectors


# Load data
image_folder = '../csc461/brn/EYE_IMAGES_FULL'
csv_path = '../csc461/brn/iris_labels_full.csv'
x_train, y_train = load_ubiris_data(image_folder, csv_path)

# Build models (encoder, decoder, autoencoder)
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Flatten()(input_img)
x = Dense(256, activation='relu')(x)
encoded = Dense(LATENT_DIM, activation='relu')(x)

decoder_input = Input(shape=(LATENT_DIM,))
x = Dense(256, activation='relu')(decoder_input)
x = Dense(IMG_SIZE * IMG_SIZE * 3, activation='sigmoid')(x)
decoded = Reshape((IMG_SIZE, IMG_SIZE, 3))(x)

encoder = Model(input_img, encoded)
decoder = Model(decoder_input, decoded)
autoencoder = Model(input_img, decoder(encoder(input_img)))

# Train
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=32,
                validation_split=0.2)


# Get class averages
class_averages = get_class_averages(encoder, x_train, y_train)

# Generate reconstructions
reconstructions = {}
for color, avg_vector in class_averages.items():
    avg_vector = avg_vector.reshape(1, LATENT_DIM)
    reconstructions[color] = decoder.predict(avg_vector)


# Create a directory to store the images if it doesn't exist
output_dir = '../csc461/brn/reconstructed_images'
os.makedirs(output_dir, exist_ok=True)


# Visualize results
plt.figure(figsize=(20, 4))
for i, (color, img) in enumerate(reconstructions.items()):
    plt.subplot(2, len(COLOR_MAPPING), i+1)
    plt.imshow(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB))
    plt.title(color)
    plt.axis('off')
    # Save the figure as a JPG file
    file_path = os.path.join(output_dir, f'{color}.jpg')
    plt.savefig(file_path)
    plt.close()  # Close the figure to release resources

plt.show()

print(f"Reconstructed images saved to '{output_dir}' directory.")

# Specify the file path
file_path = '../csc461/brn/output.txt'  # Choose your desired file name and path

# Open the file in write mode ('w')
with open(file_path, 'w') as file:
    # Write content to the file
    file.write('This is some text that will be written to the file.\n')
    # You can add more content here, e.g., variables, data, etc.
    file.write(f'The average latent vector for blue eyes is: {class_averages["blue"]}\n')
    file.write(f'The average latent vector for blue eyes is: {class_averages["brown"]}\n')
    file.write(f'The average latent vector for blue eyes is: {class_averages["hazel"]}\n')
    file.write(f'The average latent vector for blue eyes is: {class_averages["green"]}\n')
    file.write(f'The average latent vector for blue eyes is: {class_averages["gray"]}\n')


print(f'Data written to {file_path}')



