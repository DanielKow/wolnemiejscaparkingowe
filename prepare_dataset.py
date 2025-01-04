import os

base_dir = "dataset"
sub_dirs = [
    "train/images",
    "train/labels",
    "val/images",
    "val/labels"
]

for sub_dir in sub_dirs:
    directory = os.path.join(base_dir, sub_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)