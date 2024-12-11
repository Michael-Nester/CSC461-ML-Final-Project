import os
import pandas as pd

# Paths to the files and folders
original_csv_path = '../csc461/brn/iris_labels_full.csv'  # Path to the original CSV file
filtered_images_folder = '../csc461/brn/filtered_iris'  # Path to the folder with filtered images
output_csv_path = '../csc461/brn/filtered_labels.csv'  # Path to save the new CSV file

# Load the original CSV file
original_df = pd.read_csv(original_csv_path)

# Extract the list of image names from the filtered folder
filtered_image_names = set(os.listdir(filtered_images_folder))

# Ensure the image names match the format in the CSV (e.g., check for extensions)
# Assuming the image names in the CSV have extensions (e.g., 'image1.jpg')
# Adjust this if the CSV doesn't include extensions

# Get list of filtered image names without extensions
filtered_image_names = set()
for filename in os.listdir(filtered_images_folder):
    if filename.endswith('.jpg'):
        name_without_ext = os.path.splitext(filename)[0]
        filtered_image_names.add(name_without_ext)
# Filter the original dataframe to include only rows with image names in the filtered folder
filtered_df = original_df[original_df['filename'].apply(lambda x: os.path.splitext(x)[0] in filtered_image_names)]

# Save the filtered labels to a new CSV file
filtered_df.to_csv(output_csv_path, index=False)

print(f"Filtered labels saved to: {output_csv_path}")

