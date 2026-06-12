import os
import shutil
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Path to your organized dataset
source_folder = 'dataset_sequenced'
# Paths for the new data sets
train_folder = 'training_data'
val_folder = 'validation_data'
test_folder = 'test_data'

# --- Create destination folders ---
os.makedirs(os.path.join(train_folder, 'Yes'), exist_ok=True)
os.makedirs(os.path.join(train_folder, 'No'), exist_ok=True)
os.makedirs(os.path.join(val_folder, 'Yes'), exist_ok=True)
os.makedirs(os.path.join(val_folder, 'No'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'Yes'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'No'), exist_ok=True)

print("Destination folders created.")

# --- Process each class (Yes and No) ---
for class_name in ['Yes', 'No']:
    class_path = os.path.join(source_folder, class_name)
    
    # Get all the sequence folders for the current class
    sequences = os.listdir(class_path)
    print(f"Found {len(sequences)} sequences for class '{class_name}'.")

    # First split: 70% train, 30% temporary
    train_sequences, temp_sequences = train_test_split(
        sequences,
        test_size=0.3, # 30% for temp set
        random_state=42
    )

    # Second split: 50% of the temp set for validation, 50% for test (15% each of original)
    val_sequences, test_sequences = train_test_split(
        temp_sequences,
        test_size=0.5, # 50% of the temp set
        random_state=42
    )

    # --- Move files to the training folder ---
    for sequence_name in train_sequences:
        src_path = os.path.join(class_path, sequence_name)
        dest_path = os.path.join(train_folder, class_name, sequence_name)
        shutil.move(src_path, dest_path)
    
    # --- Move files to the validation folder ---
    for sequence_name in val_sequences:
        src_path = os.path.join(class_path, sequence_name)
        dest_path = os.path.join(val_folder, class_name, sequence_name)
        shutil.move(src_path, dest_path)
    
    # --- Move files to the test folder ---
    for sequence_name in test_sequences:
        src_path = os.path.join(class_path, sequence_name)
        dest_path = os.path.join(test_folder, class_name, sequence_name)
        shutil.move(src_path, dest_path)
    
    print(f"Finished splitting data for class '{class_name}'.")

print("\nData splitting complete! Your data is ready for training, validation, and testing.")