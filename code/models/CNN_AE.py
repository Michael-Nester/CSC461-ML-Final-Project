import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 128
LATENT_DIM = 64
BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLOR_MAPPING = {
    'blue': 0, 'brown': 1, 'green': 2, 'hazel': 3, 'gray': 4
}

# Add this function after the imports
def enhance_saturation(image, saturation_factor=1.5):
    """
    Convert RGB to HSV, increase saturation, convert back to RGB
    """
    # Convert to float32 for processing
    image = image.astype(np.float32) / 255.0
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Increase saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 1)
    
    # Convert back to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb

class UBIRISDataset(Dataset):
    def __init__(self, image_folder, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.df.iloc[idx]['filename'])
        image = cv2.imread(img_name)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add saturation enhancement
        #image = enhance_saturation(image)
       
        if self.transform:
            image = self.transform(image)
            
        label = COLOR_MAPPING[self.df.iloc[idx]['label']]
        return image, label

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder_linear = nn.Linear(latent_dim, 128 * 16 * 16)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.decoder_linear(x)
        x = x.view(-1, 128, 16, 16)
        return self.decoder_conv(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_model(model, train_loader, num_epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def get_class_averages(model, dataloader):
    model.eval()
    class_vectors = {color: [] for color in COLOR_MAPPING.keys()}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            encoded = model.encoder(images)
            
            for i, label in enumerate(labels):
                color = list(COLOR_MAPPING.keys())[label]
                class_vectors[color].append(encoded[i].cpu())
    
    return {color: torch.stack(vectors).mean(0) 
            for color, vectors in class_vectors.items()}

def main():
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = UBIRISDataset(
        image_folder='../csc461/brn/EYE_IMAGES_FULL',
        csv_path='../csc461/brn/iris_labels_full.csv',
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = Autoencoder(LATENT_DIM).to(DEVICE)
    
    # Train model
    train_model(model, dataloader, NUM_EPOCHS)
    
    # Get class averages
    class_averages = get_class_averages(model, dataloader)
    
    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        reconstructions = {}
        for color, avg_vector in class_averages.items():
            avg_vector = avg_vector.to(DEVICE).unsqueeze(0)
            reconstructed = model.decoder(avg_vector)
            reconstructions[color] = reconstructed.cpu().squeeze(0)
   
    output_dir = '../csc461/brn/reconstructed_images'
    os.makedirs(output_dir, exist_ok=True)


    # Visualize results
    plt.figure(figsize=(20, 4))
    for i, (color, img) in enumerate(reconstructions.items()):
        plt.subplot(2, len(COLOR_MAPPING), i+1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(color)
        plt.axis('off')
        # Save the figure as a JPG file
        file_path = os.path.join(output_dir, f'{color}.jpg')
        plt.savefig(file_path)
        plt.close()  # Close the figure to release resources

    plt.show()

    torch.save(model.state_dict(), '../csc461/brn/trained_model')

if __name__ == "__main__":
    main()
