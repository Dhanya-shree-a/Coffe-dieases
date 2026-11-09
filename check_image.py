import os
from PIL import Image

DATA_DIR = "data/all"
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

bad_images = []

for root, _, files in os.walk(DATA_DIR):
    for filename in files:
        if filename.lower().endswith(image_extensions):
            path = os.path.join(root, filename)

            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as e:
                bad_images.append((path, str(e)))

if bad_images:
    print("\nImages that cannot be read:\n")
    for path, error in bad_images:
        print(f"{path}\nError: {error}\n")
    print(f"Total invalid images: {len(bad_images)}")
else:
    print("All images passed the basic image integrity check.")
    