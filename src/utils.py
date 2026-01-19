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
    target_size=128,  # On ne donne que la taille finale souhaitée
    standardize=True,
    allowed_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF",".WEBP"),
):
    folder_path = Path(folder_path)
    images = []

    if not folder_path.exists():
        print(f"Dossier introuvable : {folder_path}")
        return images

    for file in sorted(folder_path.iterdir()):
        if not file.is_file() or file.suffix.lower() not in allowed_exts:
            continue

        try:
            # 1. Chargement
            img = Image.open(file).convert("RGB")
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = img_array.shape[:2]

            # 2. CALCUL AUTOMATIQUE DU REDIMENSIONNEMENT
            # On cherche le ratio pour que le PLUS PETIT côté soit égal à target_size
            ratio = target_size / min(h, w)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            


            # Redimensionnement proportionnel (sans distorsion)
            img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # après img_resized
            new_h, new_w = img_resized.shape[:2]

            if new_h < target_size or new_w < target_size:
                img_resized = cv2.resize(img_resized, (target_size, target_size), interpolation=cv2.INTER_AREA)
                new_h, new_w = target_size, target_size

            start_x = max(0, (new_w - target_size) // 2)
            start_y = max(0, (new_h - target_size) // 2)

            img_final = img_resized[start_y:start_y+target_size, start_x:start_x+target_size]

            # sécurité : si le crop n’est pas exactement target_size x target_size
            if img_final.shape[0] != target_size or img_final.shape[1] != target_size:
                img_final = cv2.resize(img_final, (target_size, target_size), interpolation=cv2.INTER_AREA)




            # 3. DÉCOUPE CENTRALE (CROP)
            # On calcule les coordonnées pour couper exactement au milieu
            start_x = (new_w - target_size) // 2
            start_y = (new_h - target_size) // 2
            
            img_final = img_resized[start_y:start_y+target_size, start_x:start_x+target_size]

            # 4. Standardisation (optionnel)
            if standardize:
                img_final = img_final.astype(np.float32)
                for c in range(3):
                    m, s = img_final[:,:,c].mean(), img_final[:,:,c].std()
                    img_final[:,:,c] = (img_final[:,:,c] - m) / (s + 1e-7)

            images.append((img_final, file.name))

        except Exception as e:
            print(f"Erreur sur {file.name}: {e}")

    return images