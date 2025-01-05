import os
from PIL import Image

# Define directories
train_images_dir = 'datasets/train/images'
valid_images_dir = 'datasets/valid/images'

# Target size
TARGET_SIZE = (640, 640)

def check_image_sizes(directory, target_size):
    mismatched_images = []

    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    if img.size != target_size:
                        mismatched_images.append((filename, img.size))
            except Exception as e:
                print(f"Problem z odczytem pliku '{filename}': {e}")

    return mismatched_images

train_mismatches = check_image_sizes(train_images_dir, TARGET_SIZE)
valid_mismatches = check_image_sizes(valid_images_dir, TARGET_SIZE)

# Print results
if train_mismatches or valid_mismatches:
    print("Obrazy o niewłaściwym rozmiarze:")

    if train_mismatches:
        print("\nZbiór treningowy:")
        for filename, size in train_mismatches:
            print(f"  {filename}: {size}")

    if valid_mismatches:
        print("\nZbiór walidacyjny:")
        for filename, size in valid_mismatches:
            print(f"  {filename}: {size}")
else:
    print("Wszystkie obrazy mają rozmiar 640x640.")