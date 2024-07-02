import os
import shutil
from sklearn.model_selection import train_test_split

# Path to the dataset folder
dataset_dir = '/work/cseguo/halshehri2/879/project/ISL/'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Make sure the parent train and test directories are created
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratio
test_size = 0.2  # 20% of the data will be used for testing

# Loop through each class directory in the dataset folder
for class_name in sorted(os.listdir(dataset_dir)):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Check if it's a directory and not the train/test directory
    if os.path.isdir(class_dir) and class_name not in ['train', 'test']:
        # Get all image filenames
        all_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.jpg')]
        
        # Print debugging information
        print(f"Class '{class_name}': found {len(all_files)} images")

        if len(all_files) > 0:
            # Split the data
            train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)
            
            # Create corresponding class directories in train and test folders
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            
            # Move files to their respective directories
            for f in train_files:
                shutil.move(f, train_class_dir)
            for f in test_files:
                shutil.move(f, test_class_dir)
        else:
            print(f"Warning: No images found in {class_dir}. Skipping this class.")