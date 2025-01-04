import json
import os

def convert_annotations_to_yolo(json_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    category_mapping = {
        1: 0,  # Free parking slots → YOLO class 0
        2: 1,  # Occupied parking slots → YOLO class 1
        0: -1  # Group category in coco → Log
    }

    for annotation in data.get('annotations', []):
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        yolo_category_id = category_mapping.get(category_id, -1)
        if yolo_category_id == -1:
            print(f"Category_id 0 for image_id {image_id}")

        image_info = next((img for img in data['images'] if img['id'] == image_id), None)
        if not image_info:
            continue

        img_width = image_info['width']
        img_height = image_info['height']
        file_name = image_info['file_name']

        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        yolo_line = f"{yolo_category_id} {x_center} {y_center} {norm_width} {norm_height}\n"

        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".txt")
        with open(output_file, 'a') as f:
            f.write(yolo_line)

    print(f"YOLO annotations saved in {output_folder}")

def to_yolo(category):
    json_file = f"PKLot/{category}/_annotations.coco.json"  # Replace with the path to your JSON file
    output_folder = f"PKLot/{category}/"  # Replace with the desired output folder
    convert_annotations_to_yolo(json_file, output_folder)

to_yolo("train")
to_yolo("valid")
to_yolo("test")
