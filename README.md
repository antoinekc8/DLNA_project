# Deep Learning for Soil Classification

A comprehensive deep learning project for classifying soil types using Convolutional Neural Networks (CNN), Transformer architectures, and Generative Adversarial Networks (GANs) for data augmentation.

**Team:** R. ARNAUD, M. DELPLANQUE, A. KARILA-COHEN, A. RAMPOLDI  
**Institution:** ENTPE - Mineure Data Science  
**Course:** Deep Learning for Network Analysis

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Parameters](#parameters)
- [Requirements](#requirements)

---

## Overview

This project explores various deep learning approaches for soil classification across 7 different soil types:
- **Alluvial Soil**
- **Arid Soil**
- **Black Soil**
- **Laterite Soil**
- **Mountain Soil**
- **Red Soil**
- **Yellow Soil**

We implement and compare four different approaches:
1. **Baseline CNN** - Custom convolutional neural network
2. **CNN + cGAN** - CNN with conditional GAN data augmentation
3. **CNN + StarGAN** - CNN with StarGAN data augmentation
4. **Transformer + cGAN** - Transformer with cGAN data augmentation
4. **Transformer + StarGAN** - Transformer with StarGAN data augmentation
4. **Transformer with Attention Mechanism** - Vision transformer architecture

---

## Dataset

The project uses the [Comprehensive Soil Classification Dataset](https://www.kaggle.com/datasets/ai4a-lab/comprehensive-soil-classification-datasets/code) from Kaggle.

### Data Structure

The dataset should be organized in the following structure:

```
data/
├── Original-Dataset/
│   ├── Alluvial_Soil/
│   ├── Arid_Soil/
│   ├── Black_Soil/
│   ├── Laterite_Soil/
│   ├── Mountain_Soil/
│   ├── Red_Soil/
│   └── Yellow_Soil/
└── CyAUG-Dataset/
    ├── Alluvial_Soil/
    ├── Arid_Soil/
    ├── Black_Soil/
    ├── Laterite_Soil/
    ├── Mountain_Soil/
    ├── Red_Soil/
    └── Yellow_Soil/
```

**IMPORTANT: Data Folder Placement**

The `data/` folder must be placed at the **root of the project directory** (same level as `notebooks/`, `src/`, and `results/`). The complete path should be:

```
DLNA_project/
├── data/              ← Place your data folder HERE
│   ├── Original-Dataset/
│   └── CyAUG-Dataset/
├── notebooks/
├── src/
├── results/
├── README.md
└── ...
```

---

## 📁 Project Structure

```
DLNA_project/
├── data/                           # Dataset folder (see Data Setup section)
│   ├── Original-Dataset/           # Original soil images
│   └── CyAUG-Dataset/             # Augmented dataset
├── notebooks/                      # Jupyter notebooks
│   ├── 0. Main_notebook.ipynb     # CNN baseline implementation
│   ├── 1. Transformer Attention Mechanism.ipynb
│   ├── 2. cGAN augmentation.ipynb
│   ├── 3. StarGAN augmentation.ipynb
│   └── 4. Comparison results.ipynb
├── src/                            # Source code
│   ├── __init__.py
│   ├── utils.py                   # Utility functions (data loading, preprocessing)
│   └── visualization.py           # Visualization utilities
├── results/                        # Model outputs and evaluations
│   ├── Attention/                 # Transformer results
│   ├── CNN/                       # Baseline CNN results
│   ├── GAN/                       # GAN-augmented results
│   ├── StarGAN/                   # StarGAN results
│   └── comparison/                # Comparative analysis
├── txt/
│   └── parameters.txt             # Training hyperparameters
├── pdf/                           # Documentation and reports
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA (for GPU acceleration, recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/antoinekc8/DLNA_project.git
cd DLNA_project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install torch torchvision numpy matplotlib opencv-python pillow scikit-learn jupyter
```

---

## Data Setup

### Step 1: Download the Dataset

Download the dataset from Kaggle:
- Dataset: [Comprehensive Soil Classification Dataset](https://www.kaggle.com/datasets/ai4a-lab/comprehensive-soil-classification-datasets/code)

### Step 2: Extract and Place the Data

1. Extract the downloaded dataset
2. Create a `data/` folder in the project root directory
3. Place the extracted folders inside `data/`:
   - `Original-Dataset/` - Contains the original soil images
   - `CyAUG-Dataset/` - Contains augmented images (if available)

### Step 3: Verify the Structure

Your project structure should look like this:

```
DLNA_project/
├── data/                    ← YOU MUST CREATE THIS FOLDER
│   ├── Original-Dataset/    ← PLACE DATASET HERE
│   │   ├── Alluvial_Soil/
│   │   ├── Arid_Soil/
│   │   ├── Black_Soil/
│   │   ├── Laterite_Soil/
│   │   ├── Mountain_Soil/
│   │   ├── Red_Soil/
│   │   └── Yellow_Soil/
│   └── CyAUG-Dataset/       ← PLACE AUGMENTED DATA HERE (if using)
│       └── (same structure as above)
├── notebooks/
├── src/
└── ...
```

**Note:** The notebooks expect the data to be in `../data/Original-Dataset/` or `../data/CyAUG-Dataset/` relative to the notebook location.

---

## Usage

### Running the Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` folder and run notebooks in order:
   - **0. Main_notebook.ipynb** - Start here for baseline CNN
   - **1. Transformer Attention Mechanism.ipynb** - Best performing model
   - **2. cGAN augmentation.ipynb** - Data augmentation with cGAN
   - **3. StarGAN augmentation.ipynb** - StarGAN experiments
   - **4. Comparison results.ipynb** - Compare all models

### Using Utility Functions

```python
from src.utils import load_and_pre_process_images, load_parameters

# Load parameters
params = load_parameters('txt/parameters.txt')

# Load and preprocess images
images = load_and_pre_process_images(
    folder_path='data/Original-Dataset/Alluvial_Soil',
    target_size=256,
    standardize=True
)
```

---

## Parameters

Training hyperparameters are stored in `txt/parameters.txt`:

```
TRAIN_RATIO=0.7          # Training set ratio
VAL_RATIO=0.1            # Validation set ratio
TEST_RATIO=0.2           # Test set ratio
BATCH_SIZE=64            # Batch size for training
EPOCHS=100               # Number of training epochs
LEARNING_RATE=0.01       # Initial learning rate
lr=0.0001                # Transformer learning rate
dropout=0.3              # Dropout rate
weight_decay=0.0001      # L2 regularization
SEED=42                  # Random seed for reproducibility
IMAGE_SIZE=256           # Input image size (256×256)
```

---

## Requirements

Main dependencies:
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `opencv-python` - Image processing
- `pillow` - Image handling
- `scikit-learn` - ML utilities
- `jupyter` - Interactive notebooks

---

## Notes

- All images are automatically preprocessed to 256×256 pixels with center cropping
- The preprocessing pipeline maintains aspect ratios before cropping
- Random seed (42) is set for reproducibility
- GPU acceleration is automatically used if CUDA is available
- Models are saved during training for checkpoint recovery

---

## Contributing

This is an academic project. For questions or collaboration, please contact the team members.

---

## License

This project is developed for educational purposes as part of the ENTPE Deep Learning course.

---

## Acknowledgments

- Dataset: [AI4A Lab Kaggle Dataset](https://www.kaggle.com/datasets/ai4a-lab/comprehensive-soil-classification-datasets/code)
- ENTPE - École Nationale des Travaux Publics de l'État
- Deep Learning for Network Analysis Course

---

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the team members.