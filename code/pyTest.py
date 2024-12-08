# Import required libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Load and preprocess the MNIST dataset
(x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0  # Normalize to [0, 1]
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape to (28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")


# Define the dimensions of the latent space
latent_dim = 64

# Build the Encoder
input_img = Input(shape=(28, 28, 1))
x = Flatten()(input_img)
x = Dense(128, activation='relu')(x)
latent_vector = Dense(latent_dim, activation='relu')(x)

# Build the Decoder
x = Dense(128, activation='relu')(latent_vector)
x = Dense(28 * 28, activation='sigmoid')(x)
output_img = Reshape((28, 28, 1))(x)

# Define the Autoencoder Model
autoencoder = Model(input_img, output_img)

# Compile the Model
autoencoder.compile(optimizer='adam', loss='mse')

# Summarize the model
autoencoder.summary()
