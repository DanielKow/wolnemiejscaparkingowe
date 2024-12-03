from PIL import Image
import pillow_heif
import os
import shutil

def convert_heic_to_jpg(input_path, output_path):
    heif_file = pillow_heif.open_heif(input_path)
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
    image.save(output_path, format="JPEG")
    print(f"Converted {input_path} to {output_path}")


def process_directory(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)

        if filename.lower().endswith('.heic'):
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_directory, jpg_filename)
            try:
                convert_heic_to_jpg(file_path, output_path)
                print(f"Converted {filename} to {jpg_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
        elif filename.lower().endswith('.mp4'):
            output_path = os.path.join(output_directory, filename)
            try:
                shutil.copy(file_path, output_path)
                print(f"Copied {filename} to {output_path}")
            except Exception as e:
                print(f"Failed to copy {filename}: {e}")


# Paths to directories
input_dir = 'Media_heic'
output_dir = 'Media'

# Process the directory
process_directory(input_dir, output_dir)