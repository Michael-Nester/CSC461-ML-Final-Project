




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

    def tensor_to_image(self, tensor):
        # Convert tensor to numpy array
        img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return img_array

    def save_image(self, tensor, save_path):
        # Convert tensor to image and save
        img_array = self.tensor_to_image(tensor)
        cv2.imwrite(save_path, img_array)

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
        # Load and validate image
        img = Image.open(image_path)
        if img.size[0] == 0 or img.size[1] == 0:
            raise ValueError(f"Invalid image dimensions: {img.size}")
    
        # Convert to RGB first, then grayscale for processing
        img = img.convert('RGB')
        img_gray = img.convert('L')
    
        # Transform both color and grayscale
        img_tensor = self.transform(img_gray).to(self.device)
        img_color_tensor = self.transform(img).to(self.device)
    
        # Validate tensor dimensions
        if 0 in img_tensor.shape:
            raise ValueError(f"Invalid tensor dimensions: {img_tensor.shape}")
    
        # Convert to numpy for cascade detector
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8).squeeze()
        eyes = self.cascade.detectMultiScale(img_np)
    
        detected_irises = []
        for (x, y, w, h) in eyes:
            # Validate region dimensions
            if w <= 0 or h <= 0:
                continue
            
            # Extract regions from both tensors
            eye_region_gray = img_tensor[:, y:y+h, x:x+w]
            eye_region_color = img_color_tensor[:, y:y+h, x:x+w]
        
            # Validate extracted regions
            if 0 in eye_region_gray.shape or 0 in eye_region_color.shape:
                continue
            
            # Process region
            thresh = self.threshold_image(eye_region_gray)
            processed = self.morphological_ops(thresh)
        
            # Ensure final output has correct dimensions and channels
            processed_color = eye_region_color
            if processed_color.shape[1:] != self.output_size:
                processed_color = F.resize(processed_color, self.output_size)
        
            detected_irises.append({
                'bbox': (x, y, w, h),
                'processed': processed_color
            })
    
        return detected_irises

def process_dataset(root_path, cascade_path, output_path):
    detector = IrisDetector(cascade_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    processed_images = []
    labels = []
    image_paths = []
    
    for img_path in Path(root_path).rglob('*.tiff'):
        try:
            if not os.path.getsize(img_path):  # Skip empty files
                print(f"Skipping empty file: {img_path}")
                continue
                
            irises = detector.detect_iris(str(img_path))
            
            if irises:
                for idx, iris in enumerate(irises):
                    processed = iris['processed']
                    
                    # Ensure proper channels for saving
                    if processed.dim() != 3:
                        continue
                        
                    save_path = output_path / f"{img_path.stem}_iris_{idx}.jpg"
                    detector.save_image(processed, str(save_path))
                    
                    processed_images.append(processed)
                    labels.append(img_path.parent.name)
                    image_paths.append(img_path)
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    if processed_images:
        # Verify all tensors have same size before stacking
        sizes = [img.size() for img in processed_images]
        if len(set(sizes)) > 1:
            print("Warning: Found tensors with different sizes")
            return

    # Save labels separately as CSV if needed
    if labels:
        import pandas as pd
        df = pd.DataFrame({
            'filename': [f"{path.stem}_iris_{idx}.jpg" 
                        for idx, path in enumerate(image_paths)],
            'label': labels
        })
        df.to_csv(output_path / 'labels.csv', index=False)

# Usage
if __name__ == "__main__":
    cascade_path = "../csc461/brn/haarcascade_eye.xml"
    input_path = "../csc461/brn/EYE_IMAGES_FULL"
    output_path = "../csc461/brn"
    process_dataset(input_path, cascade_path, output_path)
