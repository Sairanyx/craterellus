from pathlib import Path
import math

# Folder containing the exported YOLO label files.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
labels_dir = PROJECT_ROOT / "dataset" / "consolidated" / "labels"

# Stop if the labels folder does not exist.
if not labels_dir.is_dir():
    raise SystemExit(f"Folder not found: {labels_dir}")

# Count all annotation lines and boxes that pass every check.
total_lines = 0
valid_boxes = 0

# Find and sort all TXT files in this folder.
label_files = sorted(labels_dir.glob("*.txt"))
print(f"Found {len(label_files)} label files:")

# Process one label file at a time.
for label_file in label_files:

    # Read the file and split its text into lines.
    # utf-8-sig also handles files with a UTF-8 BOM.
    lines = label_file.read_text(encoding="utf-8-sig").splitlines()

    # Count non-empty lines, not yet validated boxes.
    object_lines = [line for line in lines if line.strip()]
    total_lines += len(object_lines)
    print(f"{label_file.name}: {len(object_lines)} annotation lines")

    # Track boxes separately for each label file.
    seen_boxes = {}
    
    # Keep the original line number for error messages.
    for line_number, line in enumerate(lines, start=1):

        # Skip empty lines.
        if not line.strip():
            continue

        # Split the line into whitespace-separated fields.
        fields = line.split()

        # Each YOLO detection line must contain exactly five fields:
        # class_id x_center y_center width height
        if len(fields) != 5:
            print(
                f"ERROR: {label_file.name}, line {line_number}: "
                f"expected 5 fields, found {len(fields)}"
            )
            continue

        # Convert all fields to numbers.
        # Report text or other values that cannot be converted.
        try:
            values = [float(field) for field in fields]
        except ValueError:
            print(
                f"ERROR: {label_file.name}, line {line_number}: "
                "non-numeric field"
            )
            continue

        # Reject NaN and positive or negative infinity.
        if not all(math.isfinite(value) for value in values):
            print(
                f"ERROR: {label_file.name}, line {line_number}: "
                "NaN or infinity is not allowed"
            )
            continue

        # Our single target category must be written as class ID 0.
        # This is a category ID, not the CNN absence label.
        if fields[0] != "0":
            print(
                f"ERROR: {label_file.name}, line {line_number}: "
                f"expected class ID 0, found {fields[0]}"
            )
            continue

                # Unpack the four bounding box values.
        _, x_center, y_center, width, height = values

        # Center coordinates must be normalized to [0, 1].
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            print(
                f"ERROR: {label_file.name}, line {line_number}: "
                "box center must be within [0, 1]"
            )
            continue

        # Width and height must be positive and no greater than 1.
        if not (0 < width <= 1 and 0 < height <= 1):
            print(
                f"ERROR: {label_file.name}, line {line_number}: "
                "width and height must be within (0, 1]"
            )
            continue

        # Calculate the four box edges in normalized coordinates.
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        # Allow tiny rounding differences in the exported values.
        tolerance = 1e-6

        # Check that all edges stay within the image boundaries.
        if (
            x_min < -tolerance
            or y_min < -tolerance
            or x_max > 1 + tolerance
            or y_max > 1 + tolerance
        ):
            print(
                f"ERROR: {label_file.name}, line {line_number}: "
                "box extends outside the image"
            )
            continue


        # Identify boxes with exactly the same numeric values.
        box_key = tuple(values)

        # Report exact duplicates, not merely overlapping boxes.
        if box_key in seen_boxes:
            print(
                f"ERROR: {label_file.name}, line {line_number}: "
                f"duplicate of line {seen_boxes[box_key]}"
            )
            continue

        # Remember the first occurrence of this box.
        seen_boxes[box_key] = line_number
        valid_boxes += 1

# Each rejected non-empty line counts as one invalid annotation.
invalid_lines = total_lines - valid_boxes

print("\nValidation summary:")
print(f"Annotation lines: {total_lines}")
print(f"Valid boxes: {valid_boxes}")
print(f"Invalid lines: {invalid_lines}")