from pathlib import Path
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def load_parameters(file_path: Path):
    """
    Load all parameters from a txt file.
    Fully evolutive: no hardcoded parameter names.
    """
    params = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            value = value.strip()

            # automatic type inference
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    pass

            params[key.strip()] = value
        
        print("\nLoaded parameters:")
        for k, v in params.items():
            print(f"  {k} = {v}")

    return params

def load_images(folder_path):
    """
    Load all images from a folder.

    Args:
        folder_path (str | Path): path to the image folder
    
    Returns:
        list: list of tuples (image_array, filename_or_path)
    """
    folder_path = Path(folder_path)
    images = []

    if not folder_path.exists():
        print(f"Folder does not exist: {folder_path}")
        return images

    for file in sorted(folder_path.iterdir()):
        if file.suffix.lower() in {".jpg", ".jpeg"}:
            try:
                img = Image.open(file).convert("RGB")
                img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                name = file.name
                images.append((img_array, name))

            except Exception as e:
                print(f"Failed to load {file.name}: {e}")

    return images
