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

    # Read and sort the CSV file according to the image naming convention
    df = pd.read_csv(labels_csv)
    df['C'] = df['filename'].str.extract(r'C(\d+)')[0].astype(int)
    df['S'] = df['filename'].str.extract(r'S(\d+)')[0].astype(int)
    df['I'] = df['filename'].str.extract(r'I(\d+)')[0].astype(int)
    df = df.sort_values(['C', 'S', 'I'])
    print(f"Total images in CSV: {len(df)}")

    # Split each class separately to ensure balanced representation
    train_dfs = []
    test_dfs = []

    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        class_train, class_test = train_test_split(
            class_df,
            test_size=test_size,
            random_state=random_state
        )
        train_dfs.append(class_train)
        test_dfs.append(class_test)

    # Combine all splits
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    
    # Sort the final datasets by filename to maintain order
    train_df = train_df.sort_values(['C', 'S', 'I'])
    test_df = test_df.sort_values(['C', 'S', 'I'])

    # Drop the temporary sorting columns
    train_df = train_df.drop(['C', 'S', 'I'], axis=1)
    test_df = test_df.drop(['C', 'S', 'I'], axis=1)

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
    SOURCE_DIR = "../csc461/brn/EYE_IMAGES_FULL"
    LABELS_CSV = "../csc461/brn/iris_labels_full.csv"
    TRAIN_DIR = "../csc461/brn/trainData"
    TEST_DIR = "../csc461/brn/testData"

    split_dataset(
        source_dir=SOURCE_DIR,
        labels_csv=LABELS_CSV,
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        test_size=0.2  # 80% train, 20% test
    )
