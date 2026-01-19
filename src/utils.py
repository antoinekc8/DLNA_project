from pathlib import Path
import os
import cv2
import numpy as np
from PIL import Image


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

def load_and_pre_process_images(
    folder_path,
    image_size=None,
    resize_to=None,
    crop_to=None,
    convert_non_jpg=True,
    standardize=False,
    allowed_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
):
    """
    Load images from a folder, optionally convert non-JPG files, resize, center-crop,
    and (optionally) standardize channels.

    Args:
        folder_path (str | Path): path to the image folder
        image_size (tuple[int, int] | None): desired final (width, height) size.
            If specified, sets both resize_to and crop_to to this value. Defaults to None.
        resize_to (tuple[int, int] | None): (width, height) to resize images before cropping.
            If None and image_size is None, images are not resized. Defaults to None.
        crop_to (tuple[int, int] | None): final (width, height) center-crop size.
            If None and image_size is None, no cropping is applied. Defaults to None.
        convert_non_jpg (bool): if True, convert non-JPG files to JPG on disk (in-place). Defaults to True.
        standardize (bool): if True, standardize channels to zero mean / unit std (float32)
        allowed_exts (tuple[str, ...]): file extensions accepted for loading

    Returns:
        list[tuple[np.ndarray, str]]: list of (image_bgr, filename)
    """
    # Use image_size for both resize and crop if specified
    if image_size is not None:
        resize_to = image_size
        crop_to = image_size
    folder_path = Path(folder_path)
    images = []

    if not folder_path.exists():
        print(f"Folder does not exist: {folder_path}")
        return images

    def _center_crop(img_arr, crop_size):
        c_w, c_h = crop_size
        start_x = max((img_arr.shape[1] - c_w) // 2, 0)
        start_y = max((img_arr.shape[0] - c_h) // 2, 0)
        end_x = start_x + c_w
        end_y = start_y + c_h
        return img_arr[start_y:end_y, start_x:end_x]

    for file in sorted(folder_path.iterdir()):
        if not file.is_file():
            continue

        ext = file.suffix.lower()
        if ext not in allowed_exts:
            if convert_non_jpg:
                try:
                    img = Image.open(file)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    new_filename = f"{file.stem}.jpg"
                    new_filepath = file.with_name(new_filename)
                    img.save(new_filepath, "JPEG", quality=95)
                    if file != new_filepath:
                        file.unlink()
                    file = new_filepath
                    ext = file.suffix.lower()
                except Exception as e:
                    print(f"Failed to convert {file.name}: {e}")
                    continue
            else:
                continue

        try:
            img = Image.open(file).convert("RGB")
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Resize to target if specified (OpenCV expects width, height)
            if resize_to is not None:
                r_w, r_h = resize_to
                if (img_array.shape[1], img_array.shape[0]) != (r_w, r_h):
                    img_array = cv2.resize(img_array, (r_w, r_h), interpolation=cv2.INTER_AREA)

            # Center-crop if specified
            if crop_to is not None:
                img_array = _center_crop(img_array, crop_to)

            # Optional standardization (channel-wise)
            if standardize:
                img_std = np.zeros_like(img_array, dtype=np.float32)
                for c in range(img_array.shape[2]):
                    channel = img_array[:, :, c].astype(np.float32)
                    mean = np.mean(channel)
                    std = np.std(channel)
                    if std > 0:
                        img_std[:, :, c] = (channel - mean) / std
                    else:
                        img_std[:, :, c] = channel - mean
                img_array = img_std

            images.append((img_array, file.name))

        except Exception as e:
            print(f"Failed to load {file.name}: {e}")

    return images
