import os
from PIL import Image

# Define directories
train_dir = 'datasets/train/images'
valid_dir = 'datasets/valid/images'


def get_image_sizes(directory):
    """Get sizes of all images in a given directory."""
    image_sizes = {}
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add supported extensions
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    image_sizes[filename] = img.size  # Get the (width, height) tuple
            except Exception as e:
                print(f"Error reading image {filename}: {e}")
    return image_sizes


def find_differing_sizes(train_sizes, valid_sizes):
    """Find images with differing sizes within or across datasets."""
    differing_images = []

    # Check train images only
    train_size_set = set(train_sizes.values())
    train_diff = [name for name, size in train_sizes.items() if size not in train_size_set]

    # Check valid images only
    valid_size_set = set(valid_sizes.values())
    valid_diff = [name for name, size in valid_sizes.items() if size not in valid_size_set]

    # Combine the differing information
    differing_images.extend(train_diff)
    differing_images.extend(valid_diff)

    return differing_images


# Get image sizes
train_sizes = get_image_sizes(train_dir)
valid_sizes = get_image_sizes(valid_dir)

# Check for differing sizes
differing_images = find_differing_sizes(train_sizes, valid_sizes)

# Print results
if differing_images:
    print("Images with differing sizes:")
    for img in differing_images:
        print(img)
else:
    print("All images have the same sizes.")