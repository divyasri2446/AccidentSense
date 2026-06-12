import os
import shutil
import pandas as pd

# Define paths
csv_file_path = 'Crash_Table.csv'
source_image_folder = 'CrashBest'
destination_base_folder = 'dataset_sequenced'

# Get sorted list of image files
try:
    image_files = sorted(os.listdir(source_image_folder))
    print(f"Found {len(image_files)} image files.")
except FileNotFoundError:
    print(f"Error: Folder '{source_image_folder}' not found. Exiting.")
    exit()

# Get the labels
try:
    df = pd.read_csv(csv_file_path)
    labels = df['egoinvolve'].tolist()
    print(f"Found {len(labels)} labels in the CSV.")
except FileNotFoundError:
    print(f"Error: CSV file '{csv_file_path}' not found. Exiting.")
    exit()

# Create destination folders for Yes and No labels
yes_folder = os.path.join(destination_base_folder, 'Yes')
no_folder = os.path.join(destination_base_folder, 'No')
os.makedirs(yes_folder, exist_ok=True)
os.makedirs(no_folder, exist_ok=True)

# Group and move files
frames_per_event = len(image_files) // len(labels)
print(f"Grouping {len(image_files)} images into {len(labels)} events, with {frames_per_event} frames per event.")

for i, label in enumerate(labels):
    start_index = i * frames_per_event
    end_index = start_index + frames_per_event
    
    # Determine destination folder
    dest_folder = yes_folder if label == 'Yes' else no_folder
    
    # Create a unique subfolder for this sequence
    sequence_folder_name = f'sequence_{i:04d}'
    sequence_path = os.path.join(dest_folder, sequence_folder_name)
    os.makedirs(sequence_path, exist_ok=True)
    
    # Move the files for this sequence
    for j in range(start_index, end_index):
        if j < len(image_files):
            image_name = image_files[j]
            source_path = os.path.join(source_image_folder, image_name)
            dest_path = os.path.join(sequence_path, image_name)
            shutil.move(source_path, dest_path)
            
print("\nData organization complete!")