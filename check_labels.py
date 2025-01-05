import os

train_labels_dir = 'datasets/train/labels'
valid_labels_dir = 'datasets/valid/labels'

def validate_label_rows(directory):
    invalid_rows = []

    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue
        
        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, 'r') as file:
                for line_number, line in enumerate(file, start=1):
                    values = line.strip().split()

                    if not values:
                        invalid_rows.append((filename, line_number, "Pusty wiersz"))
                        continue

                    try:
                        first_num = int(values[0])
                        if first_num not in {0, 1}:
                            invalid_rows.append(
                                (filename, line_number, f"Kategoria'{first_num}' nie jest 0 ani 1"))
                    except ValueError:
                        invalid_rows.append(
                            (filename, line_number, f"Nieporawna kategoria '{values[0]}'"))
                        continue

                    # Validate the remaining numbers are floats between 0 and 1
                    for i, size in enumerate(values[1:], start=2):  # Start at 2 for field position clarity
                        try:
                            size_float = float(size)
                            if not (0 <= size_float <= 1):
                                invalid_rows.append((filename, line_number,
                                                     f"Wartość '{size_float}' na pozycji {i} spoza  zakresu 0 i 1"))
                        except ValueError:
                            invalid_rows.append(
                                (filename, line_number, f"Nieporawna wartość '{size}' na pozycji {i}"))

        except Exception as e:
            invalid_rows.append((filename, -1, f"Bład podczas odczytu pliku: {str(e)}"))

    return invalid_rows


train_invalid = validate_label_rows(train_labels_dir)
valid_invalid = validate_label_rows(valid_labels_dir)

if train_invalid or valid_invalid:
    print("Wyniki walidacji etykiet:")
    if train_invalid:
        print("\nNiepoprawne etykiety w zbiorze treningowym:")
        for file, row, problem in train_invalid:
            print(f"  Plik: {file}, Wiersz: {row}, Problem: {problem}")

    if valid_invalid:
        print("\nNieporawne etykiety w zbiorze walidacyjnym:")
        for file, row, problem in valid_invalid:
            print(f"  Plik: {file}, Wiersz: {row}, Problem: {problem}")
else:
    print("Wszystkie etykiety są poprawne :)")