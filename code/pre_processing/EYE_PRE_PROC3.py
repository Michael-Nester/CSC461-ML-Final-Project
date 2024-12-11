import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
import cv2

class IrisDetector:
    def __init__(self, cascade_path, output_size=(28, 28), device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.output_size = output_size


        # Define transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((400, 300))
        ])
        
        # Create morphological kernels as tensors
        self.kernel = torch.ones(5, 5).to(self.device)
    
    def normalize_size(self, tensor):
        return F.resize(tensor.unsqueeze(0), self.output_size).squeeze(0)

    def threshold_image(self, img_tensor, threshold=0):
        if threshold == 0:
            # Otsu's thresholding
            val = torch.quantile(img_tensor, 0.5)
            thresh = (img_tensor > val).float()
        else:
            thresh = (img_tensor > threshold).float()
        return thresh
    
    def morphological_ops(self, thresh_tensor):
        # Convert operations to PyTorch
        padding = 2
        padded = F.pad(thresh_tensor, (padding, padding, padding, padding))
        
        # Opening
        eroded = torch.nn.functional.max_pool2d(padded, 5, stride=1)
        opening = torch.nn.functional.conv2d(eroded.unsqueeze(0), 
                                           self.kernel.unsqueeze(0).unsqueeze(0),
                                           padding=padding)
        
        # Closing
        dilated = torch.nn.functional.conv2d(padded.unsqueeze(0),
                                           self.kernel.unsqueeze(0).unsqueeze(0),
                                           padding=padding)
        closing = torch.nn.functional.max_pool2d(dilated, 5, stride=1)
        
        # Combine
        result = torch.logical_or(opening, closing).float()
        return result.squeeze()
    
    def detect_iris(self, image_path):
        # Load image
        img = Image.open(image_path).convert('L')  # Grayscale
        img_tensor = self.transform(img).to(self.device)
        
        # Convert to numpy for cascade detector
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8).squeeze()
        eyes = self.cascade.detectMultiScale(img_np)
        
        detected_irises = []
        for (x, y, w, h) in eyes:
            # Extract eye region
            eye_region = img_tensor[:, y:y+h, x:x+w]
            
            # Process eye region
            thresh = self.threshold_image(eye_region)
            processed = self.morphological_ops(thresh)
            
            detected_irises.append({
                'bbox': (x, y, w, h),
                'processed': processed
            })
        
        return detected_irises

def process_dataset(root_path, cascade_path, output_path):
    detector = IrisDetector(cascade_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    processed_images = []
    labels = []
    
    for img_path in Path(root_path).rglob('*.tiff'):
        try:
            # Process image
            irises = detector.detect_iris(str(img_path))
            
            if irises:
                # Save processed image
                for idx, iris in enumerate(irises):
                    processed = iris['processed']

                    # Verify tensor size
                    if processed.size() != (detector.output_size[0], detector.output_size[1]):
                        processed = detector.normalize_size(processed)

                    save_path = output_path / f"{img_path.stem}_iris_{idx}.pt"
                    torch.save(processed, save_path)
                    
                    processed_images.append(processed)
                    labels.append(img_path.parent.name)  # Assuming folder name is label
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if processed_images:
        # Verify all tensors have same size before stacking
        sizes = [img.size() for img in processed_images]
        if len(set(sizes)) > 1:
            print("Warning: Found tensors with different sizes")
            return

        # Save dataset
        torch.save({
            'images': torch.stack(processed_images),
            'labels': labels
        }, output_path / 'processed_dataset.pt')

# Usage
if __name__ == "__main__":
    cascade_path = "../csc461/brn/haarcascade_eye.xml"
    input_path = "../csc461/brn/EYE_IMAGES_FULL"
    output_path = "../csc461/brn"
    process_dataset(input_path, cascade_path, output_path)
