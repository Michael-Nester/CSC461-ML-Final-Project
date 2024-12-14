import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(
    source_dir,
    labels_csv,
    train_dir,
    test_dir,
    test_size=0.2,
    random_state=42
):
    """
    Split dataset into train and test sets.
    
    Args:
        source_dir (str): Directory containing all .tiff images
        labels_csv (str): Path to CSV file with image names and labels
        train_dir (str): Directory to store training images
        test_dir (str): Directory to store test images
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducibility
    """
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(labels_csv)
    print(f"Total images in CSV: {len(df)}")
    
    # Split the DataFrame
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # Ensure balanced split across colors
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Function to copy images and handle errors
    def copy_images(df, destination_dir, split_name):
        successful_copies = []
        for idx, row in df.iterrows():
            src_path = os.path.join(source_dir, row['filename'])
            dst_path = os.path.join(destination_dir, row['filename'])
            
            try:
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    successful_copies.append(row)
                else:
                    print(f"Warning: Source file not found: {src_path}")
            except Exception as e:
                print(f"Error copying {row['filename']}: {str(e)}")
        
        # Create DataFrame with only successful copies
        return pd.DataFrame(successful_copies)
    
    # Copy images and create new CSVs
    print("\nCopying training images...")
    train_df = copy_images(train_df, train_dir, "training")
    
    print("Copying test images...")
    test_df = copy_images(test_df, test_dir, "test")
    
    # Save the new CSVs
    train_csv_path = os.path.join(os.path.dirname(train_dir), 'train_labels.csv')
    test_csv_path = os.path.join(os.path.dirname(test_dir), 'test_labels.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    # Print summary
    print("\nDataset split complete!")
    print(f"Training images saved to: {train_dir}")
    print(f"Test images saved to: {test_dir}")
    print(f"Training labels saved to: {train_csv_path}")
    print(f"Test labels saved to: {test_csv_path}")
    
    # Print distribution of colors in both sets
    print("\nColor distribution in training set:")
    print(train_df['label'].value_counts())
    print("\nColor distribution in test set:")
    print(test_df['label'].value_counts())

# Example usage
if __name__ == "__main__":
    # Set your paths here
    SOURCE_DIR = "path/to/your/tiff/images"
    LABELS_CSV = "path/to/your/labels.csv"
    TRAIN_DIR = "path/to/output/train"
    TEST_DIR = "path/to/output/test"
    
    split_dataset(
        source_dir=SOURCE_DIR,
        labels_csv=LABELS_CSV,
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        test_size=0.2  # 80% train, 20% test
    )
