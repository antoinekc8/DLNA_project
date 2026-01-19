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

def load_images(folder_path, resize_to=(160, 160), crop_to=(128, 128)):
    """
    Load all images from a folder and preprocess to a fixed size by:
    1) resizing to `resize_to` (default 160x160), then
    2) center-cropping to `crop_to` (default 128x128).

    Args:
        folder_path (str | Path): path to the image folder
        resize_to (tuple[int, int]): (width, height) to resize images to before cropping. Defaults to (160, 160).
        crop_to (tuple[int, int]): final (width, height) center-crop size. Defaults to (128, 128).
    
    Returns:
        list[tuple[np.ndarray, str]]: list of tuples (image_array_bgr, filename)
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
                # Step 1: resize to resize_to (OpenCV expects (width, height))
                r_w, r_h = resize_to
                if (img_array.shape[1], img_array.shape[0]) != (r_w, r_h):
                    img_array = cv2.resize(img_array, (r_w, r_h), interpolation=cv2.INTER_AREA)

                # Step 2: center-crop to crop_to
                c_w, c_h = crop_to
                start_x = max((img_array.shape[1] - c_w) // 2, 0)
                start_y = max((img_array.shape[0] - c_h) // 2, 0)
                end_x = start_x + c_w
                end_y = start_y + c_h
                img_array = img_array[start_y:end_y, start_x:end_x]
                name = file.name
                images.append((img_array, name))

            except Exception as e:
                print(f"Failed to load {file.name}: {e}")

    return images
