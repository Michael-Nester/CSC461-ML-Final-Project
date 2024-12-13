import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np



class EyeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            return None

def load_data(image_dir, labels_csv):
    """
    Load image paths and their corresponding eye color labels from CSV
    
    Args:
        image_dir (str): Directory containing eye images
        labels_csv (str): Path to CSV file containing image names and eye color labels
    """
    # Read the CSV file
    df = pd.read_csv(labels_csv)
    
    # Create dictionary to map eye colors to numerical labels
    unique_colors = df['label'].unique()
    color_to_idx = {color: idx for idx, color in enumerate(unique_colors)}
    
    image_paths = []
    labels = []
    
    # Match images with their labels
    for _, row in df.iterrows():
        image_name = row['filename']  # Adjust column name as per your CSV
        eye_color = row['label']    # Adjust column name as per your CSV
        
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(color_to_idx[eye_color])
    
    return image_paths, labels, color_to_idx

def get_data_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    }

def setup_model(num_classes, device):
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    
    # Modify the final layer for eye color classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    model = model.to(device)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, color_to_idx):
    best_val_acc = 0.0
    idx_to_color = {v: k for k, v in color_to_idx.items()}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        class_correct = {color: 0 for color in color_to_idx.keys()}
        class_total = {color: 0 for color in color_to_idx.keys()}
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for label, pred in zip(labels, predicted):
                    color = idx_to_color[label.item()]
                    class_total[color] += 1
                    if label == pred:
                        class_correct[color] += 1
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for color in color_to_idx.keys():
            if class_total[color] > 0:
                accuracy = 100. * class_correct[color] / class_total[color]
                print(f'{color}: {accuracy:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'color_to_idx': color_to_idx,
                'best_accuracy': best_val_acc
            }, '../csc461/brn/best_eye_color_classifier.pth')
        
        print('-' * 60)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    image_dir = "../csc461/brn/EYE_IMAGES_FULL"
    labels_csv = "../csc461/brn/iris_labels_full.csv"

    # Load data
    image_paths, labels, color_to_idx = load_data(image_dir, labels_csv)
    num_classes = len(color_to_idx)
    print(f"Number of classes: {num_classes}")
    print("Eye colors:", list(color_to_idx.keys()))

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # Hyperparameters
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    transforms_dict = get_data_transforms()
    
    train_dataset = EyeDataset(train_paths, train_labels, transform=transforms_dict['train'])
    val_dataset = EyeDataset(val_paths, val_labels, transform=transforms_dict['val'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model, criterion, and optimizer
    model = setup_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, color_to_idx)

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torchvision.models as models
