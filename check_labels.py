import os

train_labels_dir = 'datasets/train/labels'
valid_labels_dir = 'datasets/valid/labels'

def validate_label_rows(directory):
    invalid_rows = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, 'r') as file:
                    for line_number, line in enumerate(file, start=1):
                        values = line.strip().split()

                        # Ensure the row isn't empty
                        if not values:
                            invalid_rows.append((filename, line_number, "Empty row"))
                            continue

                        # Validate the first number is 0 or 1
                        try:
                            first_num = int(values[0])
                            if first_num not in {0, 1}:
                                invalid_rows.append(
                                    (filename, line_number, f"First number '{first_num}' is not 0 or 1"))
                        except ValueError:
                            invalid_rows.append(
                                (filename, line_number, f"First number '{values[0]}' is not valid integer"))
                            continue

                        # Validate the remaining numbers are floats between 0 and 1
                        for i, size in enumerate(values[1:], start=2):  # Start at 2 for field position clarity
                            try:
                                size_float = float(size)
                                if not (0 <= size_float <= 1):
                                    invalid_rows.append((filename, line_number,
                                                         f"Value '{size_float}' at position {i} not between 0 and 1"))
                            except ValueError:
                                invalid_rows.append(
                                    (filename, line_number, f"Value '{size}' at position {i} not a valid float"))

            except Exception as e:
                invalid_rows.append((filename, -1, f"Error reading file: {str(e)}"))

    return invalid_rows


train_invalid = validate_label_rows(train_labels_dir)
valid_invalid = validate_label_rows(valid_labels_dir)

if train_invalid or valid_invalid:
    print("Label validation results:")
    if train_invalid:
        print("\nInvalid rows in train labels:")
        for file, row, problem in train_invalid:
            print(f"  File: {file}, Row: {row}, Problem: {problem}")

    if valid_invalid:
        print("\nInvalid rows in valid labels:")
        for file, row, problem in valid_invalid:
            print(f"  File: {file}, Row: {row}, Problem: {problem}")
else:
    print("All label rows in train and valid datasets are valid.")