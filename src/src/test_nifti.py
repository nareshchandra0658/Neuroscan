
from pathlib import Path
from PIL import Image
import numpy as np

# Actual dataset path found in your project
DATA_DIR = Path(
    r"D:\NEUROSCAN\NEUROSCAN\data\Epic and CSCR hospital Dataset"
)

# Standard class names for both datasets
EXPECTED_CLASSES = {
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

print("NeuroScan Dataset Checker")
print("Dataset path:", DATA_DIR)
print("Dataset exists:", DATA_DIR.exists())

if not DATA_DIR.exists():
    print("ERROR: Dataset folder not found.")

else:
    for split in ["Train", "Test"]:

        split_path = DATA_DIR / split

        print("\n" + "=" * 45)
        print(f"{split.upper()} DATASET")
        print("=" * 45)

        if not split_path.is_dir():
            print("Missing folder:", split_path)
            continue

        # Find class folders directly inside Train or Test
        class_folders = {
            folder.name.lower(): folder
            for folder in split_path.iterdir()
            if folder.is_dir()
        }

        total_images = 0

        print("\nClass folder validation:")

        for class_name in sorted(EXPECTED_CLASSES):
            if class_name in class_folders:
                print(f"[FOUND] {class_name}")
            else:
                print(f"[MISSING] {class_name}")

        # Inspect all top-level folders, including unexpected ones
        for folder_name, folder_path in sorted(class_folders.items()):

            image_files = [
                file for file in folder_path.rglob("*")
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            ]

            total_images += len(image_files)

            print(f"\nFolder: {folder_name}")
            print("Image count:", len(image_files))

            if image_files:
                sample_path = image_files[0]

                try:
                    with Image.open(sample_path) as image:
                        image.load()
                        pixels = np.asarray(image.convert("RGB"))

                    print("Sample:", sample_path.name)
                    print("Image dimensions:", pixels.shape)
                    print("Data type:", pixels.dtype)
                    print("Minimum intensity:", int(pixels.min()))
                    print("Maximum intensity:", int(pixels.max()))
                    print("PASS: Image loaded successfully.")

                except Exception as error:
                    print("Image loading error:", error)

        print(f"\nTotal {split} images: {total_images}")

print("\nDataset inspection completed.")
