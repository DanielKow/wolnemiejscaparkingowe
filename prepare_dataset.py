import os
import shutil
from pathlib import Path

base_dir = "datasets"

shutil.rmtree(base_dir)

sub_dirs = [
    "train/images",
    "train/labels",
    "valid/images",
    "valid/labels"
]

for sub_dir in sub_dirs:
    directory = os.path.join(base_dir, sub_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def copy_images_and_labels(category):
    source_directory = f"PKLot/{category}/"
    for image in Path(source_directory).glob("*.jpg"):
        shutil.copy(image, f"{base_dir}/{category}/images")
        label = image.with_suffix(".txt")
        if not label.exists():
            label.touch()
        shutil.copy(label, f"{base_dir}/{category}/labels")

copy_images_and_labels("train")
copy_images_and_labels("valid")