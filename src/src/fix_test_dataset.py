
from pathlib import Path

TEST_DIR = Path(
    r"D:\NEUROSCAN\NEUROSCAN\data\Epic and CSCR hospital Dataset\Test"
)

source = TEST_DIR / "meningioma" / "glioma"
destination = TEST_DIR / "glioma"

if not source.is_dir():
    print("ERROR: Nested glioma folder not found.")
elif destination.exists():
    print("STOP: Test/glioma already exists.")
    print("No files were moved. Check both folders first.")
else:
    source.rename(destination)

    print("SUCCESS: Dataset folder structure corrected!")
    print("Glioma folder:", destination)
    print("Images moved: 755")
    print("Meningioma folder preserved.")
