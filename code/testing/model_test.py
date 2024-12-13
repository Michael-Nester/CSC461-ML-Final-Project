import torch
from CNN_AE import AutoEncoder  # Replace with your autoencoder class and file
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Load the autoencoder model
model = AutoEncoder()
model.load_state_dict(torch.load(""))
model.eval()

# Load the input eye image
eye_image = Image.open("../home/csc461/brn/eye_image.jpg")  # Replace with your image path

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),  # Match your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match your training normalization
])

# Preprocess the image
input_tensor = preprocess(eye_image).unsqueeze(0)  # Add batch dimension

# Get the reconstruction
with torch.no_grad():
    reconstructed_image = model(input_tensor)

# Convert reconstructed image to numpy
reconstructed_image_np = reconstructed_image.squeeze(0).permute(1, 2, 0).numpy()
reconstructed_image_np = ((reconstructed_image_np + 1) * 127.5).astype(np.uint8)  # Denormalize

# Convert to BGR for OpenCV
reconstructed_image_bgr = cv2.cvtColor(reconstructed_image_np, cv2.COLOR_RGB2BGR)

# Create an iris mask (assume circular iris in the center)
height, width, _ = reconstructed_image_bgr.shape
center = (width // 2, height // 2)
radius = min(height, width) // 4  # Adjust based on your images

mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(mask, center, radius, (255), thickness=-1)  # Create circular mask

# Blend blue hue with the iris
blue_color = np.array([255, 0, 0], dtype=np.uint8)  # Blue in BGR for OpenCV
blue_layer = np.zeros_like(reconstructed_image_bgr)
blue_layer[:] = blue_color

# Apply the mask to the blue layer
iris_hue = cv2.bitwise_and(blue_layer, blue_layer, mask=mask)
reconstructed_with_hue = cv2.addWeighted(reconstructed_image_bgr, 1.0, iris_hue, 0.5, 0)

# Convert back to RGB and save
final_image = cv2.cvtColor(reconstructed_with_hue, cv2.COLOR_BGR2RGB)
final_pil_image = Image.fromarray(final_image)
final_pil_image.save("../home/csc/461/output_eye_with_blue_hue.jpg")
final_pil_image.show()

