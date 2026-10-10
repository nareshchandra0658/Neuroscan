
from pathlib import Path

DATA_DIR = Path(
    r"D:\NEUROSCAN\NEUROSCAN\data\Epic and CSCR hospital Dataset"
)

TEST_DIR = DATA_DIR / "Test"

print("Searching for glioma folders inside Test...\n")

for folder in TEST_DIR.rglob("*"):
    if folder.is_dir() and "glioma" in folder.name.lower():
        images = [
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
            }
        ]

        print("Folder:", folder)
        print("Image count:", len(images))
        print("-" * 40)

print("\nSearch completed.")
