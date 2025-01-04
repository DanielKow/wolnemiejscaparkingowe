import json
import os

def convert_annotations_to_yolo(json_file, output_folder):
    """
    Convert PKLot JSON annotations to YOLO format.
    
    Args:
        json_file (str): Path to the JSON annotation file.
        output_folder (str): Path to save YOLO annotations.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Parse categories to create a mapping (e.g., {"spaces": 0, "space-empty": 1, "space-occupied": 2})
    category_mapping = {category['name']: category['id'] for category in data['categories']}

    # Iterate through the annotations
    for annotation in data.get('annotations', []):
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        # Find the corresponding image details
        image_info = next((img for img in data['images'] if img['id'] == image_id), None)
        if not image_info:
            continue

        img_width = image_info['width']
        img_height = image_info['height']
        file_name = image_info['file_name']

        # Convert bbox to YOLO format
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        # YOLO annotation line
        yolo_line = f"{category_id} {x_center} {y_center} {norm_width} {norm_height}\n"

        # Save YOLO annotations in a text file with the same name as the image
        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".txt")
        with open(output_file, 'a') as f:
            f.write(yolo_line)

    print(f"YOLO annotations saved in {output_folder}")

# Example Usage

def to_yolo(category):
    json_file = f"PKLot/{category}/_annotations.coco.json"  # Replace with the path to your JSON file
    output_folder = f"PKLot/{category}/"  # Replace with the desired output folder
    convert_annotations_to_yolo(json_file, output_folder)

to_yolo("train")
to_yolo("valid")
to_yolo("test")
