from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image


ParamsDict = Dict[str, object]
ImageEntry = Tuple[np.ndarray, str]


def load_parameters(file_path: Path) -> ParamsDict:
    """Load parameters from a simple key=value text file with basic casting."""

    params: ParamsDict = {}

    with open(file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            lowered = value.lower()
            if lowered in {"true", "false"}:
                params[key] = lowered == "true"
                continue

            try:
                params[key] = float(value) if "." in value else int(value)
            except ValueError:
                params[key] = value

    print("\nLoaded parameters:")
    for name, val in params.items():
        print(f"  {name} = {val}")

    return params


def _center_crop_square(image: np.ndarray, size: int) -> np.ndarray:
    """Return a centered square crop of the requested size, padding via resize if needed."""

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Received empty image")

    if height < size or width < size:
        # If either side is smaller, pad by resizing after crop attempt
        scale = size / min(height, width)
        new_w = max(int(round(width * scale)), size)
        new_h = max(int(round(height * scale)), size)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]

    start_y = max((height - size) // 2, 0)
    start_x = max((width - size) // 2, 0)

    end_y = start_y + size
    end_x = start_x + size

    if end_y > height:
        start_y = max(height - size, 0)
        end_y = start_y + size
    if end_x > width:
        start_x = max(width - size, 0)
        end_x = start_x + size

    cropped = image[start_y:end_y, start_x:end_x]

    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

    return cropped


def _standardize_per_channel(image: np.ndarray) -> np.ndarray:
    """Standardize each channel with safeguards against tiny standard deviations."""

    standardized = image.astype(np.float32, copy=False)
    for channel in range(standardized.shape[2]):
        plane = standardized[:, :, channel]
        mean = plane.mean()
        std = plane.std()
        standardized[:, :, channel] = (plane - mean) / (std + 1e-7)
    return standardized


def load_and_pre_process_images(
    folder_path: Union[Path, str],
    *,
    target_size: int = 128,
    standardize: bool = True,
    allowed_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
) -> List[ImageEntry]:
    """Load, resize, center-crop to a square, and optionally standardize images."""

    folder = Path(folder_path)
    images: List[ImageEntry] = []

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return images

    allowed = {ext.lower() for ext in allowed_exts}

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in allowed:
            continue

        try:
            rgb = Image.open(file_path).convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

            scale = target_size / min(bgr.shape[:2])
            new_w = max(int(round(bgr.shape[1] * scale)), target_size)
            new_h = max(int(round(bgr.shape[0] * scale)), target_size)
            resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cropped = _center_crop_square(resized, target_size)
            if standardize:
                cropped = _standardize_per_channel(cropped)

            images.append((cropped, file_path.name))

        except Exception as exc:
            print(f"Error processing {file_path.name}: {exc}")

    return images