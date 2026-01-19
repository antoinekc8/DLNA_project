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

            # 3. DÉCOUPE CENTRALE (CROP)
            # On calcule les coordonnées pour couper exactement au milieu
            start_x = (new_w - target_size) // 2
            start_y = (new_h - target_size) // 2
            
            img_final = img_resized[start_y:start_y+target_size, start_x:start_x+target_size]

         
            # ... (après le crop)
            img_final = img_resized[start_y:start_y+target_size, start_x:start_x+target_size]

            # FORCE la taille exacte au cas où l'arrondi aurait échoué
            if img_final.shape[0] != target_size or img_final.shape[1] != target_size:
                img_final = cv2.resize(img_final, (target_size, target_size))

            # 4. Standardisation
            if standardize:
                img_final = img_final.astype(np.float32)
                # On normalise globalement plutôt que par canal si on veut éviter les divisions par zéro bizarres
                m, s = img_final.mean(), img_final.std()
                img_final = (img_final - m) / (s + 1e-7)

            images.append((img_final, file.name))



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



import warnings
warnings.filterwarnings('ignore')  # Suppress warnings to keep notebook clean

import random
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import sys
import pandas as pd
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset




PROJECT_ROOT = Path.cwd().parent.resolve()
DATA_DIR= PROJECT_ROOT / "data"
PARAM_FILE = PROJECT_ROOT / "txt" / "parameters.txt"
# utils.py functions
UTILS_DIR = PROJECT_ROOT / "src"
sys.path.append(str(PROJECT_ROOT / "src"))
from utils import load_parameters, load_and_pre_process_images
from visualization import show_soil_grid


DEVICE = 'cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, "xpu") and torch.xpu.is_available() else 'cpu'
print(f"Params loaded. Device: {DEVICE}")



# Load parameters from external file
params = load_parameters(PARAM_FILE)
globals().update(params)
soil_types = params["SOIL_TYPES"].split(",")



# Use parameters for seed and device
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



# Load and preprocess images
TARGET_SIZE = 256  # Example target size, adjust as needed
images_dict = {}  # Dictionary to hold images per soil type
for soil in soil_types:



    folder = DATA_DIR / "Orignal-Dataset" / soil
    images = load_and_pre_process_images(
        folder_path=folder,
        target_size=TARGET_SIZE  # <--- Indispensable
    )
    images_dict[soil] = images
    print(f"{soil}: {len(images)} images loaded and preprocessed")



    # images_dict = {soil_type: [(img_bgr, filename), ...], ...}
show_soil_grid(images_dict, n_per_type=5, tile_size=(240, 240), pad=12)






# Create results folder structure
from pathlib import Path

results_root = (Path('..') / 'results').resolve()
subfolders = ['CNN', 'GAN', 'StarGAN', 'Attention']

results_root.mkdir(parents=True, exist_ok=True)
for name in subfolders:
    (results_root / name).mkdir(parents=True, exist_ok=True)

print(f"Created/verified results root: {results_root}")
for name in subfolders:
    print(f" - {results_root / name}")



    import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        # 1. 32 filtres | Entrée 256x256 -> Sortie 128x128 (après pool)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 2. 64 filtres | 128x128 -> 64x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 3. 128 filtres | 64x64 -> 32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 4. 256 filtres | 32x32 -> 16x16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 5. 512 filtres | 16x16 -> 8x8
        # Pour du 256x256, une 5ème couche permet d'extraire des features très riches
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Entrée 512 car conv5 sort 512 canaux
        self.fc1 = nn.Linear(512, 256)
        self.drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # Global Average Pooling : (batch, 512, 8, 8) -> (batch, 512, 1, 1)
        x = self.gap(x)
        x = x.view(x.size(0), -1) 

        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)
    


    all_images = []
all_labels = []
for soil_type, images in images_dict.items():
    for img, filename in images:
        all_images.append(img)
        all_labels.append(soil_type)

X_temp, X_test, y_temp, y_test = train_test_split(all_images, all_labels, test_size=TEST_RATIO, random_state=SEED, stratify=all_labels)
relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=relative_val_ratio, random_state=SEED, stratify=y_temp)

print(f"Dataset split: train ({len(X_train)}), val ({len(X_val)}), test ({len(X_test)})")









import numpy as np
from sklearn.model_selection import train_test_split

# 1. On prépare deux listes vides
all_images = []
all_labels = []

# On définit l'ordre des classes (0 à 6)
class_names = sorted(images_dict.keys()) 

for idx, soil in enumerate(class_names):
    for img_data, img_name in images_dict[soil]:
        all_images.append(img_data) # L'image (256, 256, 3)
        all_labels.append(idx)      # Le numéro de la classe (0-6)

# 2. LE SPLIT (80% train, 20% val)
# On utilise sklearn car c'est plus simple pour les listes numpy
X_train, X_val, y_train, y_val = train_test_split(
    all_images, 
    all_labels, 
    test_size=0.2, 
    random_state=42,
    stratify=all_labels # Très important : garde les mêmes proportions de classes
)

# 3. Création du Dataset PyTorch compatible
class SoilDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # On force la conversion en float32 et le format CHW (3, 256, 256)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label

# 4. On crée les objets pour les Dataloaders
train_ds = SoilDataset(X_train, y_train)
val_ds = SoilDataset(X_val, y_val)


class SoilDataset(Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels
        unique_labels = sorted(list(set(labels)))
        self.soil_type_to_idx = {soil_type: idx for idx, soil_type in enumerate(unique_labels)}
        self.label_indices = [self.soil_type_to_idx[label] for label in labels]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):         
        img = self.data[idx]
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.FloatTensor(img)
        label = self.label_indices[idx]
        return img_tensor, label

train_dataset = SoilDataset(X_train, y_train)
val_dataset = SoilDataset(X_val, y_val)
test_dataset = SoilDataset(X_test, y_test)
print(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


model = SimpleCNN().to(DEVICE)  # Create model and move to device

loss_fn = nn.CrossEntropyLoss()  # Binary Cross Entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # Adam optimizer using weight decay which helps regularization in order to reduce overfitting
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Test with our blueprint model
from util_2 import inspect_architecture
inspect_architecture(model, "SimpleCNN")



def train_with_validation(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=100):
    """Train the model with validation monitoring and save best checkpoint."""
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()  # Reset gradients
            preds = model(xb)      # Forward pass
            loss = loss_fn(preds, yb)  # Compute loss
            loss.backward()        # Backward pass
            optimizer.step()       # Update weights using the gradients

            train_loss += loss.item()
            correct += (preds.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item()
                correct += (preds.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            # Save best model checkpoint to ../model/best_model
            from pathlib import Path
            model_dir = (Path('..') / 'model').resolve()
            model_dir.mkdir(parents=True, exist_ok=True)
            save_path = model_dir / 'best_model'
            torch.save(best_state, save_path)
            print(f"Saved new best model to: {save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, train_losses, val_losses, train_accs, val_accs


model, train_losses, val_losses, train_accs, val_accs = train_with_validation(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=DEVICE,
    epochs=EPOCHS
)

