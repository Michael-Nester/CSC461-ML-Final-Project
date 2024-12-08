import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np




class EnhancedTraining:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.losses = {
            'zero_one': [],
            'l2': [],
            'absolute': [],
            'train_loss': [],
            'val_loss': []
        }

    def find_lr(self, start_lr=1e-4, end_lr=1e-2, num_iterations=100):
        """Find optimal learning rate using learning rate range test"""
        current_lr = start_lr
        mult = (end_lr / start_lr) ** (1/num_iterations)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)
        lrs = []
        losses = []
        
        print("Finding optimal learning rate...")
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if batch_idx > num_iterations:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            lrs.append(current_lr)
            losses.append(loss.item())
            
            current_lr *= mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # Plot learning rate vs loss
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        plt.savefig('lr_finder.png')
        plt.close()
        
        # Find the learning rate with steepest descent
        optimal_lr = lrs[np.argmin(losses)] / 10  # Divide by 10 for safety
        print(f"Optimal learning rate: {optimal_lr:.2e}")
        return optimal_lr

    def calculate_losses(self, outputs, targets):
        """Calculate different types of losses"""
        # 0/1 Loss
        _, predicted = outputs.max(1)
        zero_one_loss = (predicted != targets).float().mean().item()
        
        # L2 Loss (MSE)
        l2_loss = nn.MSELoss()(outputs, nn.functional.one_hot(targets, num_classes=outputs.size(1)).float()).item()
        
        # Absolute Loss (L1)
        absolute_loss = nn.L1Loss()(outputs, nn.functional.one_hot(targets, num_classes=outputs.size(1)).float()).item()
        
        return zero_one_loss, l2_loss, absolute_loss

    def train_model(self, epochs=100, patience=10):
        """Train model with early stopping and learning rate scheduling"""
        optimal_lr = self.find_lr()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=optimal_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model = None
        
        # Initialize loss tracking
        with open('../csc461/brn/losses.txt', 'w') as f:
            f.write("Epoch,Train_Loss,Val_Loss,Zero_One_Loss,L2_Loss,Absolute_Loss\n")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            all_zero_one = []
            all_l2 = []
            all_absolute = []
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate additional losses
                zero_one, l2, absolute = self.calculate_losses(outputs, targets)
                all_zero_one.append(zero_one)
                all_l2.append(l2)
                all_absolute.append(absolute)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            val_loss = self.validate()
            scheduler.step(val_loss)
            
            # Average losses
            avg_train_loss = train_loss / len(self.train_loader)
            avg_zero_one = np.mean(all_zero_one)
            avg_l2 = np.mean(all_l2)
            avg_absolute = np.mean(all_absolute)
            
            # Save losses
            with open('../csc461/brn/losses.txt', 'a') as f:
                f.write(f"{epoch},{avg_train_loss},{val_loss},{avg_zero_one},{avg_l2},{avg_absolute}\n")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Restore best model
        self.model.load_state_dict(best_model)
        return self.model

    def validate(self):
        """Perform validation"""
        self.model.eval()
        val_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        return val_loss / len(self.val_loader)

    def save_model(self, path, color_to_idx):
        """Save model with additional metadata"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'color_to_idx': color_to_idx,
            'losses': self.losses
        }, path)

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
    """
    Load image paths and their corresponding eye color labels from CSV
    """
    # Read the CSV file
    df = pd.read_csv(labels_csv)
    print("CSV loaded with shape:", df.shape)

    # Get list of all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tiff')])
    print(f"Found {len(image_files)} images in directory")

    # Create lists for paths and labels
    image_paths = []
    labels = []

    # Create label mapping
    unique_colors = sorted(df['label'].unique())
    color_to_idx = {color: idx for idx, color in enumerate(unique_colors)}
    print("Color mapping:", color_to_idx)

    # Match images with labels
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        # Find corresponding label in DataFrame
        label_row = df[df['filename'] == img_file]

        if not label_row.empty:
            image_paths.append(img_path)
            labels.append(color_to_idx[label_row['label'].iloc[0]])

    print(f"Successfully matched {len(image_paths)} images with labels")
    print(f"Number of unique labels: {len(color_to_idx)}")

    if len(image_paths) == 0 or len(labels) == 0:
        raise ValueError("No images were matched with labels. Check file names and CSV content.")

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
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze most of the layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Only unfreeze the last few layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Add feature extraction method
    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    model.get_features = get_features.__get__(model)
    
    # Modify the final layer with proper initialization
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Reduced dropout
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    # Initialize the new layers properly
    nn.init.xavier_uniform_(model.fc[1].weight)
    nn.init.zeros_(model.fc[1].bias)
    nn.init.xavier_uniform_(model.fc[4].weight)
    nn.init.zeros_(model.fc[4].bias)
    
    return model.to(device)

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
            }, 'best_eye_color_classifier2.pth')

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

    # Create datasets and dataloaders
    transforms_dict = get_data_transforms()


    train_dataset = EyeDataset(train_paths, train_labels, transform=transforms_dict['train'])
    val_dataset = EyeDataset(val_paths, val_labels, transform=transforms_dict['val'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    num_classes = len(color_to_idx)
    model = setup_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize EnhancedTraining
    trainer = EnhancedTraining(model, train_loader, val_loader, device)
        # Train with enhanced features
    model = trainer.train_model(epochs=100, patience=10)
    
    # Save the model with additional metadata
    trainer.save_model('best_eye_color_classifier3.pth', color_to_idx)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(full_dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")

    # Print label distribution
    label_counts = {}
    for _, label in full_dataset:
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel distribution:")
    idx_to_label = {v: k for k, v in full_dataset.label_to_idx.items()}
    for idx, count in label_counts.items():
        print(f"{idx_to_label[idx]}: {count} images")


if __name__ == "__main__":
    main()
