import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix


def inspect_architecture(model, model_name="Model"):
    """Inspect and display architecture details"""
    print(f"\n=== {model_name} Architecture Inspection ===")
    print("=" * 50)
    
    total_params = 0
    trainable_params = 0
    
    for i, (name, layer) in enumerate(model.named_modules()):
        if len(list(layer.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in layer.parameters())
            total_params += params
            
            if layer.parameters():
                trainable_params += params
                print(f"{i+1:2d}. {name:20s} | {type(layer).__name__:15s} | Params: {params:8d}")
    
    print("=" * 50)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    return total_params


def plot_regression_predictions(model, data_loader, device, title: str, ax=None):
    model.eval()
    xs, ys, preds = [], [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            pred = model(xb.to(device)).cpu().numpy()
            xs.append(xb.numpy())
            ys.append(yb.numpy())
            preds.append(pred)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    order = np.argsort(x_all[:, 0])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.scatter(x_all, y_all, s=16, alpha=0.5, label="true")
    ax.plot(x_all[order], p_all[order], color="red", lw=2, label="pred")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 15)
    ax.set_title(title)
    ax.legend()

    if ax is None:
        plt.show()

def get_mnist_transforms():
    """Get standard MNIST transforms with normalization"""
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


class MNIST50Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Each folder name (00000, 00001, etc.) corresponds to the digit class
        for class_folder in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path) and class_folder.isdigit():
                class_label = int(class_folder)  # Convert folder name to class label
                
                # Load all PNG files in this class folder
                for filename in os.listdir(class_path):
                    if filename.endswith('.png'):
                        self.samples.append({
                            'path': os.path.join(class_path, filename),
                            'label': class_label
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']



def create_dataloaders(root_dir='datasets/mnist_50', 
                                train_ratio=0.7, 
                                val_ratio=0.1, 
                                batch_size=32, 
                                seed=42,
                                n_classes=10):
    """
    Create train/val/test dataloaders with equal samples per class.
    
    Args:
        root_dir: Path to dataset directory
        train_ratio: Training set ratio (default 0.7)
        val_ratio: Validation set ratio (default 0.1) 
        batch_size: Batch size for dataloaders
        seed: Random seed for reproducibility
        n_classes: Number of classes (default 10 for MNIST)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load dataset with transforms
    transform = get_mnist_transforms()
    mnist_dataset = MNIST50Dataset(root_dir=root_dir, transform=transform)
    
    # Group indices by class
    indices_by_class = {i: [] for i in range(n_classes)}
    for idx, sample in enumerate(mnist_dataset.samples):
        indices_by_class[sample['label']].append(idx)
    
    # Stratified split
    g = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = [], [], []
    
    for c in range(n_classes):
        idxs = torch.tensor(indices_by_class[c])
        n = len(idxs)  # expected 50
        
        # Calculate exact counts per class
        n_train = int(n * train_ratio)  # 35
        n_val = int(n * val_ratio)      # 5
        n_test = n - n_train - n_val    # 10
        assert n_train + n_val + n_test == n
        
        # Random permutation and split
        perm = torch.randperm(n, generator=g)
        idxs = idxs[perm]
        train_indices.extend(idxs[:n_train].tolist())
        val_indices.extend(idxs[n_train:n_train+n_val].tolist())
        test_indices.extend(idxs[n_train+n_val:].tolist())
    
    # Create subsets and dataloaders
    train_dataset = Subset(mnist_dataset, train_indices)
    val_dataset = Subset(mnist_dataset, val_indices)
    test_dataset = Subset(mnist_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def verify_class_distribution(train_loader, val_loader, test_loader, n_classes=10):
    """
    Print class distribution across train/val/test sets
    
    Args:
        train_loader, val_loader, test_loader: DataLoaders to verify
        n_classes: Number of classes (default 10)
    """
    train_class_counts = {i: 0 for i in range(n_classes)}
    val_class_counts = {i: 0 for i in range(n_classes)}
    test_class_counts = {i: 0 for i in range(n_classes)}
    
    # Count classes in each split
    for _, label in train_loader.dataset:
        train_class_counts[label] += 1
    for _, label in val_loader.dataset:
        val_class_counts[label] += 1
    for _, label in test_loader.dataset:
        test_class_counts[label] += 1
    
    # Print results
    total_train = len(train_loader.dataset)
    total_val = len(val_loader.dataset)
    total_test = len(test_loader.dataset)
    
    print(f"Dataset sizes: Train={total_train}, Val={total_val}, Test={total_test}")
    print(f"Total images: {total_train + total_val + total_test}")
    
    print("\nPer-class distribution:")
    print("Class | Train | Val | Test")
    print("------|-------|-----|-----")
    for i in range(n_classes):
        print(f"  {i}   |  {train_class_counts[i]:2d}   | {val_class_counts[i]:2d}  |  {test_class_counts[i]:2d}")
        


def train_with_validation(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=100, task_type='classification'):
    """
    Train the model with validation monitoring during training.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to run on
        epochs: Number of epochs
        task_type: 'classification' or 'regression'
    
    Returns:
        For classification: (model, train_losses, val_losses, train_accuracies, val_accuracies)
        For regression: (model, train_losses, val_losses, train_metrics, val_metrics)
    """
        
    train_losses = []
    val_losses = []
    train_metrics = []  # Renamed for clarity
    val_metrics = []    # Renamed for clarity
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_actuals = []
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            
            # Handle shape mismatch for regression
            if task_type == 'regression' and preds.dim() > yb.dim():
                preds = preds.squeeze()
            
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if task_type == 'classification':
                _, predicted = preds.max(1)
                train_total += yb.size(0)
                train_correct += predicted.eq(yb).sum().item()
            else:  # regression
                train_preds.extend(preds.detach().cpu().numpy())
                train_actuals.extend(yb.cpu().numpy())
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        if task_type == 'classification':
            train_acc = 100. * train_correct / train_total
            train_metrics.append(train_acc)
        else:  # regression - calculate RMSE
            train_rmse = np.sqrt(mean_squared_error(train_actuals, train_preds))
            train_metrics.append(train_rmse)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_actuals = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                
                # Handle shape mismatch for regression
                if task_type == 'regression' and preds.dim() > yb.dim():
                    preds = preds.squeeze()
                
                val_loss += loss_fn(preds, yb).item()
                
                if task_type == 'classification':
                    _, predicted = preds.max(1)
                    val_total += yb.size(0)
                    val_correct += predicted.eq(yb).sum().item()
                else:  # regression
                    val_preds.extend(preds.cpu().numpy())
                    val_actuals.extend(yb.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if task_type == 'classification':
            val_acc = 100. * val_correct / val_total
            val_metrics.append(val_acc)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Training Loss: {train_loss:.4f} | Training Acc: {train_acc:.2f}% | "
                      f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")
        else:  # regression
            val_rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
            val_metrics.append(val_rmse)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Training Loss: {train_loss:.6f} | Training RMSE: {train_metrics[-1]:.6f} | "
                      f"Validation Loss: {val_loss:.6f} | Validation RMSE: {val_rmse:.6f}")
    
    return model, train_losses, val_losses, train_metrics, val_metrics

def test_model(model, test_loader, loss_fn, device, task_type='classification'):
    """
    Perform final testing on the model using the held-out test set.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        loss_fn: Loss function (CrossEntropyLoss for classification, MSELoss for regression)
        device: Device to run on
        task_type: 'classification' or 'regression'
    
    Returns:
        For classification: (test_loss, test_accuracy, predictions, actual_values)
        For regression: (test_loss, rmse, predictions, actual_values)
    """
    
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_actuals = []
    
    if task_type == 'classification':
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                test_loss += loss_fn(preds, yb).item()
                
                _, predicted = preds.max(1)
                test_total += yb.size(0)
                test_correct += predicted.eq(yb).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_actuals.extend(yb.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
        
        return test_loss, test_acc, all_preds, all_actuals
    
    elif task_type == 'regression':
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                
                # Handle different output shapes for time series models
                if preds.dim() > 1 and preds.size(1) == 1:
                    preds = preds.squeeze()
                
                test_loss += loss_fn(preds, yb).item()
                
                all_preds.extend(preds.cpu().numpy())
                all_actuals.extend(yb.cpu().numpy())
        
        test_loss /= len(test_loader)
        rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
        
        return test_loss, rmse, all_preds, all_actuals
    
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")



### Enhanced plotting function for training and validation results
def plo_training_results_with_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot comprehensive training results including loss, accuracy, and convergence analysis.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    # Plot training curves
    plt.figure(figsize=(10, 6))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Additional training analysis
    print("Training Results Analysis:")
    print(f"  Final training loss: {train_losses[-1]:.4f}")
    print(f"  Final validation loss: {val_losses[-1]:.4f}")
    print(f"  Final training accuracy: {train_accuracies[-1]:.2f}%")
    print(f"  Final validation accuracy: {val_accuracies[-1]:.2f}%")


def plot_detailed_training_analysis(train_losses, val_losses, train_accuracies, val_accuracies, 
                                  window_size=3, trend_window=5, smoothness_threshold_low=0.1, 
                                  smoothness_threshold_high=0.5, figure_size=(10, 5), alpha=0.6, 
                                  linewidth=2, grid_alpha=0.3):
    """
    Plot detailed training analysis with moving averages and comprehensive statistics.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        window_size: Window size for moving average (default: 3)
        trend_window: Number of epochs to compare for trend analysis (default: 5)
        smoothness_threshold_low: Lower threshold for smoothness analysis (default: 0.1)
        smoothness_threshold_high: Upper threshold for smoothness analysis (default: 0.5)
        figure_size: Figure size as tuple (width, height) (default: (10, 5))
        alpha: Transparency for raw data lines (default: 0.6)
        linewidth: Line width for moving average lines (default: 2)
        grid_alpha: Grid transparency (default: 0.3)

    """
    plt.figure(figsize=figure_size)

    # Calculate moving averages
    train_loss_ma = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
    val_loss_ma = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')

    # Loss with moving average
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, alpha=alpha, label='Train Loss (raw)', color='lightblue')
    plt.plot(val_losses, alpha=alpha, label='Val Loss (raw)', color='lightcoral')
    plt.plot(range(window_size-1, len(train_losses)), train_loss_ma, 
             label='Train Loss (MA)', color='blue', linewidth=linewidth)
    plt.plot(range(window_size-1, len(val_losses)), val_loss_ma, 
             label='Val Loss (MA)', color='red', linewidth=linewidth)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves with Moving Average')
    plt.legend()
    plt.grid(True, alpha=grid_alpha)

    # Accuracy with moving average
    train_acc_ma = np.convolve(train_accuracies, np.ones(window_size)/window_size, mode='valid')
    val_acc_ma = np.convolve(val_accuracies, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, alpha=alpha, label='Train Acc (raw)', color='lightblue')
    plt.plot(val_accuracies, alpha=alpha, label='Val Acc (raw)', color='lightcoral')
    plt.plot(range(window_size-1, len(train_accuracies)), train_acc_ma, 
             label='Train Acc (MA)', color='blue', linewidth=linewidth)
    plt.plot(range(window_size-1, len(val_accuracies)), val_acc_ma, 
             label='Val Acc (MA)', color='red', linewidth=linewidth)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves with Moving Average')
    plt.legend()
    plt.grid(True, alpha=grid_alpha)

    plt.tight_layout()
    plt.show()

    # Detailed statistical analysis
    print("Detailed Loss and Accuracy Analysis:")
    print(f"  Moving average window size: {window_size}")

    # Loss statistics
    print(f"\nLoss Statistics:")
    print(f"  Training loss - Mean: {np.mean(train_losses):.4f}, Std: {np.std(train_losses):.4f}")
    print(f"  Validation loss - Mean: {np.mean(val_losses):.4f}, Std: {np.std(val_losses):.4f}")

    # Accuracy statistics
    print(f"\nAccuracy Statistics:")
    print(f"  Training accuracy - Mean: {np.mean(train_accuracies):.2f}%, Std: {np.std(train_accuracies):.2f}%")
    print(f"  Validation accuracy - Mean: {np.mean(val_accuracies):.2f}%, Std: {np.std(val_accuracies):.2f}%")

    # Trend analysis
    if len(train_losses) > trend_window:
        early_train_loss = np.mean(train_losses[:trend_window])
        late_train_loss = np.mean(train_losses[-trend_window:])
        early_val_loss = np.mean(val_losses[:trend_window])
        late_val_loss = np.mean(val_losses[-trend_window:])
        
        print(f"\nTrend Analysis (first {trend_window} vs last {trend_window} epochs):")
        print(f"  Training loss: {early_train_loss:.4f} → {late_train_loss:.4f} (change: {late_train_loss - early_train_loss:.4f})")
        print(f"  Validation loss: {early_val_loss:.4f} → {late_val_loss:.4f} (change: {late_val_loss - early_val_loss:.4f})")
        
        if late_train_loss < early_train_loss and late_val_loss < early_val_loss:
            print(f"  Both training and validation improved over time")
        elif late_train_loss < early_train_loss and late_val_loss >= early_val_loss:
            print(f"  Training improved but validation may have overfit")
        else:
            print(f"  Training may need more epochs or different approach")

    # Smoothness analysis
    print(f"\nSmoothness Analysis:")
    train_loss_smoothness = np.std(np.diff(train_losses))
    val_loss_smoothness = np.std(np.diff(val_losses))

    print(f"  Training loss smoothness (std of differences): {train_loss_smoothness:.4f}")
    print(f"  Validation loss smoothness (std of differences): {val_loss_smoothness:.4f}")

    if train_loss_smoothness < smoothness_threshold_low and val_loss_smoothness < smoothness_threshold_low:
        print(f"  Training appears stable and smooth")
    elif train_loss_smoothness > smoothness_threshold_high or val_loss_smoothness > smoothness_threshold_high:
        print(f"  Training appears unstable (high variance between epochs)")

    print(f"\nDetailed loss and accuracy analysis completed!")

    # # Return analysis results for programmatic use
    # return {
    #     'parameters': {
    #         'window_size': window_size,
    #         'trend_window': trend_window,
    #         'smoothness_threshold_low': smoothness_threshold_low,
    #         'smoothness_threshold_high': smoothness_threshold_high
    #     },
    #     'loss_stats': {
    #         'train_mean': np.mean(train_losses),
    #         'train_std': np.std(train_losses),
    #         'val_mean': np.mean(val_losses),
    #         'val_std': np.std(val_losses)
    #     },
    #     'accuracy_stats': {
    #         'train_mean': np.mean(train_accuracies),
    #         'train_std': np.std(train_accuracies),
    #         'val_mean': np.mean(val_accuracies),
    #         'val_std': np.std(val_accuracies)
    #     },
    #     'smoothness': {
    #         'train_loss': train_loss_smoothness,
    #         'val_loss': val_loss_smoothness
    #     },
    #     'moving_averages': {
    #         'train_loss_ma': train_loss_ma,
    #         'val_loss_ma': val_loss_ma,
    #         'train_acc_ma': train_acc_ma,
    #         'val_acc_ma': val_acc_ma
    #     }
    # }




def analyze_training_convergence(train_accuracies, val_accuracies, window_size=10, convergence_threshold=0.5):
    """
    Analyze training convergence based on accuracy variance in recent epochs.
    
    Args:
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch  
        window_size: Number of recent epochs to analyze (default: 10)
        convergence_threshold: Standard deviation threshold for convergence (default: 0.5)
    
    Returns:
        dict: Dictionary containing convergence analysis results
    """
    if len(train_accuracies) <= window_size:
        print(f"Not enough epochs for convergence analysis (need > {window_size} epochs)")
        return None
    
    # Get recent accuracies
    recent_train_acc = train_accuracies[-window_size:]
    recent_val_acc = val_accuracies[-window_size:]
    
    # Calculate standard deviations
    train_acc_std = np.std(recent_train_acc)
    val_acc_std = np.std(recent_val_acc)
    
    # Determine convergence status
    has_converged = train_acc_std < convergence_threshold and val_acc_std < convergence_threshold
    
    # Print analysis
    print(f"\nConvergence Analysis (last {window_size} epochs):")
    print(f"  Training accuracy std: {train_acc_std:.3f}")
    print(f"  Validation accuracy std: {val_acc_std:.3f}")
    
    if has_converged:
        print(f"  Model appears to have converged (low variance in recent epochs)")
    else:
        print(f"  Model may still be learning (high variance in recent epochs)")
    
    print(f"\nTraining visualization completed!")
    
    # Return results for programmatic use
    return {
        'train_acc_std': train_acc_std,
        'val_acc_std': val_acc_std,
        'has_converged': has_converged,
        'window_size': window_size,
        'convergence_threshold': convergence_threshold
    }



def analyze_classification_detailed(targets, predictions, n_classes=10, top_confused_pairs=5, 
                                  show_classification_report=True):
    """
    Detailed classification analysis with statistics and metrics.
    
    Args:
        targets: True labels
        predictions: Predicted labels
        n_classes: Number of classes (default: 10)
        top_confused_pairs: Number of top confused pairs to show (default: 5)
        show_classification_report: Whether to show sklearn classification report (default: True)
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Create confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Print classification report
    if show_classification_report:
        print("Classification Report:")
        print(classification_report(targets, predictions, target_names=[str(i) for i in range(n_classes)]))
    
    # Detailed confusion matrix analysis
    print(f"\nConfusion Matrix Analysis:")
    print(f"  Total samples: {len(targets)}")
    print(f"  Correct predictions: {np.sum(predictions == targets)}")
    print(f"  Incorrect predictions: {np.sum(predictions != targets)}")
    print(f"  Overall accuracy: {100 * np.sum(predictions == targets) / len(targets):.2f}%")
    
    # Most confused pairs
    print(f"\nMost Confused Digit Pairs:")
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((i, j, cm[i, j]))
    
    # Sort by confusion count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, count in confusion_pairs[:top_confused_pairs]:
        print(f"  {i} → {j}: {count} times")
    
    # Per-class detailed analysis
    print(f"\nPer-Class Detailed Analysis:")
    for class_label in range(n_classes):
        true_pos = cm[class_label, class_label]
        false_pos = cm[:, class_label].sum() - true_pos
        false_neg = cm[class_label, :].sum() - true_pos
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Class {class_label}:")
        print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"    True Pos: {true_pos}, False Pos: {false_pos}, False Neg: {false_neg}")
    
    print(f"\nDetailed classification analysis completed!")



def analyze_classification_detailed(targets, predictions, n_classes=10, top_confused_pairs=5, 
                                  show_classification_report=True):
    """
    Detailed classification analysis with statistics and metrics.
    
    Args:
        targets: True labels
        predictions: Predicted labels
        n_classes: Number of classes (default: 10)
        top_confused_pairs: Number of top confused pairs to show (default: 5)
        show_classification_report: Whether to show sklearn classification report (default: True)
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Create confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Print classification report
    if show_classification_report:
        print("Classification Report:")
        print(classification_report(targets, predictions, target_names=[str(i) for i in range(n_classes)]))
    
    # Detailed confusion matrix analysis
    print(f"\nConfusion Matrix Analysis:")
    print(f"  Total samples: {len(targets)}")
    print(f"  Correct predictions: {np.sum(predictions == targets)}")
    print(f"  Incorrect predictions: {np.sum(predictions != targets)}")
    print(f"  Overall accuracy: {100 * np.sum(predictions == targets) / len(targets):.2f}%")
    
    # Most confused pairs
    print(f"\nMost Confused Digit Pairs:")
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((i, j, cm[i, j]))
    
    # Sort by confusion count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, count in confusion_pairs[:top_confused_pairs]:
        print(f"  {i} → {j}: {count} times")
    
    # Per-class detailed analysis
    print(f"\nPer-Class Detailed Analysis:")
    for class_label in range(n_classes):
        true_pos = cm[class_label, class_label]
        false_pos = cm[:, class_label].sum() - true_pos
        false_neg = cm[class_label, :].sum() - true_pos
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Class {class_label}:")
        print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"    True Pos: {true_pos}, False Pos: {false_pos}, False Neg: {false_neg}")
    
    print(f"\nDetailed classification analysis completed!")



def plot_confusion_matrix(targets, predictions, n_classes=10, figure_size=(10, 8), 
                         cmap='Blues', annot_fmt='d', title='Confusion Matrix',
                         normalize=False, annot_fmt_norm='.2f'):
    """
    Plot confusion matrix with optional normalization.
    
    Args:
        targets: True labels
        predictions: Predicted labels
        n_classes: Number of classes (default: 10)
        figure_size: Figure size (default: (10, 8))
        cmap: Colormap (default: 'Blues')
        annot_fmt: Annotation format for raw matrix (default: 'd')
        title: Plot title (default: 'Confusion Matrix')
        normalize: Whether to normalize the confusion matrix (default: False)
        annot_fmt_norm: Annotation format for normalized matrix (default: '.2f')
    """

    
    # Create confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = annot_fmt_norm
        if 'Normalized' not in title:
            title = f"{title} (Normalized)"
    else:
        cm_display = cm
        fmt = annot_fmt
    
    # Plot confusion matrix
    plt.figure(figsize=figure_size)
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()




def visualize_misclassified_examples(model, test_loader, device, num_examples=12, figsize=(15, 10)):
    """
    Visualize misclassified examples from the test set.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run model on
        num_examples: Maximum number of misclassified examples to show
        figsize: Figure size for the plot
    
    Returns:
        tuple: (misclassified_indices, misclassified_images, true_labels, predicted_labels)
    """
    model.eval()
    
    # Collect all test data
    all_images = []
    all_true_labels = []
    all_predicted_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_images.extend(images.cpu())
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Find misclassified examples
    misclassified_mask = all_true_labels != all_predicted_labels
    misclassified_indices = np.where(misclassified_mask)[0]
    
    print(f"Found {len(misclassified_indices)} misclassified examples")
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return None, None, None, None
    
    # Limit the number of examples to display
    num_to_show = min(num_examples, len(misclassified_indices))
    selected_indices = misclassified_indices[:num_to_show]
    
    # Create subplots
    rows = (num_to_show + 3) // 4  # 4 columns
    cols = min(4, num_to_show)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot misclassified examples
    for i, idx in enumerate(selected_indices):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Get image and labels
        image = all_images[idx].squeeze()  # Remove channel dimension for grayscale
        true_label = all_true_labels[idx]
        pred_label = all_predicted_labels[idx]
        confidence = all_probabilities[idx][pred_label]
        
        # Display image
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}', 
                    fontsize=10, color='red')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_to_show, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Misclassified Examples (showing {num_to_show} of {len(misclassified_indices)})', 
                 fontsize=14, y=1.02)
    plt.show()
    
    # Analyze misclassification patterns
    print(f"\nMisclassification Analysis:")
    print(f"Most common misclassification patterns:")
    
    # Count misclassification patterns
    pattern_counts = {}
    for idx in misclassified_indices:
        true_label = all_true_labels[idx]
        pred_label = all_predicted_labels[idx]
        pattern = f"{true_label} → {pred_label}"
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Sort by frequency
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    for pattern, count in sorted_patterns[:5]:  # Show top 5 patterns
        print(f"  {pattern}: {count} times")
    
    print(f"\nMisclassified examples visualization completed!")
    
    return (misclassified_indices, 
            [all_images[i] for i in misclassified_indices], 
            [all_true_labels[i] for i in misclassified_indices],
            [all_predicted_labels[i] for i in misclassified_indices])


def analyze_roc_performance(roc_auc, roc_auc_micro, roc_auc_macro, n_classes=10):
    """
    Analyze and interpret ROC curve performance results.
    
    Args:
        roc_auc: Dictionary of per-class AUC scores
        roc_auc_micro: Micro-average AUC score
        roc_auc_macro: Macro-average AUC score
        n_classes: Number of classes (default: 10)
    
    Returns:
        dict: Analysis results including performance metrics and interpretations
    """
    
    print("="*60)
    print("ROC PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Basic metrics
    print(f"Overall Performance Metrics:")
    print(f"  Micro-average AUC: {roc_auc_micro:.3f}")
    print(f"  Macro-average AUC: {roc_auc_macro:.3f}")
    print(f"  Mean per-class AUC: {np.mean(list(roc_auc.values())):.3f}")
    print(f"  Std per-class AUC: {np.std(list(roc_auc.values())):.3f}")
    
    # Per-class detailed results
    print(f"\nPer-class AUC scores:")
    for i in range(n_classes):
        performance_level = ""
        if roc_auc[i] >= 0.95:
            performance_level = "(Excellent)"
        elif roc_auc[i] >= 0.85:
            performance_level = "(Good)"
        elif roc_auc[i] >= 0.75:
            performance_level = "(Fair)"
        else:
            performance_level = "(Needs Improvement)"
        print(f"  Digit {i}: {roc_auc[i]:.3f} {performance_level}")
    
    # Ranking analysis
    print(f"\nPerformance Ranking:")
    best_classes = sorted([(i, roc_auc[i]) for i in range(n_classes)], 
                         key=lambda x: x[1], reverse=True)
    
    print(f"  Top 3 performing classes:")
    for i, (class_idx, auc_score) in enumerate(best_classes[:3]):
        print(f"    {i+1}. Digit {class_idx}: AUC = {auc_score:.3f}")
    
    print(f"  Bottom 3 performing classes:")
    for i, (class_idx, auc_score) in enumerate(best_classes[-3:]):
        rank = len(best_classes) - 2 + i
        print(f"    {rank}. Digit {class_idx}: AUC = {auc_score:.3f}")
    
    # Overall interpretation
    print(f"\nOverall Performance Interpretation:")
    if roc_auc_micro > 0.95 and roc_auc_macro > 0.95:
        overall_performance = "Excellent"
        interpretation = "Outstanding classification performance across all classes"
    elif roc_auc_micro > 0.9 and roc_auc_macro > 0.9:
        overall_performance = "Very Good"
        interpretation = "Very good classification performance across all classes"
    elif roc_auc_micro > 0.8 and roc_auc_macro > 0.8:
        overall_performance = "Good"
        interpretation = "Good classification performance across all classes"
    elif roc_auc_micro > 0.7 and roc_auc_macro > 0.7:
        overall_performance = "Fair"
        interpretation = "Fair classification performance - room for improvement"
    else:
        overall_performance = "Poor"
        interpretation = "Classification performance needs significant improvement"
    
    print(f"  Performance Level: {overall_performance}")
    print(f"  Summary: {interpretation}")
    
    # Variance analysis
    auc_std = np.std(list(roc_auc.values()))
    print(f"\nClass Performance Consistency:")
    print(f"  Standard deviation: {auc_std:.3f}")
    
    if auc_std > 0.1:
        consistency_level = "High variance"
        consistency_advice = "Significant differences between classes - investigate class-specific issues"
    elif auc_std > 0.05:
        consistency_level = "Moderate variance"
        consistency_advice = "Some differences between classes - consider class-specific improvements"
    else:
        consistency_level = "Low variance"
        consistency_advice = "Consistent performance across classes"
    
    print(f"  Consistency Level: {consistency_level}")
    print(f"  Recommendation: {consistency_advice}")
    
    # Micro vs Macro comparison
    print(f"\nMicro vs Macro AUC Analysis:")
    auc_diff = abs(roc_auc_micro - roc_auc_macro)
    print(f"  Difference: {auc_diff:.3f}")
    
    if auc_diff < 0.02:
        balance_assessment = "Well-balanced performance"
        balance_advice = "Model performs consistently across all classes"
    elif auc_diff < 0.05:
        balance_assessment = "Slightly imbalanced performance"
        balance_advice = "Minor class imbalance effects observed"
    else:
        balance_assessment = "Imbalanced performance"
        balance_advice = "Significant class imbalance effects - consider rebalancing strategies"
    
    print(f"  Assessment: {balance_assessment}")
    print(f"  Implication: {balance_advice}")
    
    # Specific recommendations
    print(f"\nRecommendations:")
    
    # Find problematic classes
    poor_classes = [i for i in range(n_classes) if roc_auc[i] < 0.8]
    if poor_classes:
        print(f"  - Focus on improving classes: {poor_classes}")
        print(f"    These classes have AUC < 0.8 and need attention")
    
    # General recommendations based on performance
    if overall_performance in ["Poor", "Fair"]:
        print(f"  - Consider increasing model complexity or training time")
        print(f"  - Review data quality and feature engineering")
        print(f"  - Try different architectures or hyperparameters")
    elif overall_performance == "Good":
        print(f"  - Fine-tune hyperparameters for better performance")
        print(f"  - Consider ensemble methods")
    else:
        print(f"  - Performance is excellent - model is ready for deployment")
        print(f"  - Consider model compression or optimization for production")
    
    if auc_std > 0.05:
        print(f"  - Investigate class-specific data distribution")
        print(f"  - Consider class-specific data augmentation")
        print(f"  - Review confusion matrix for specific error patterns")
    
    print("="*60)
    
    # Return analysis results for programmatic use
    return {
        'overall_performance': overall_performance,
        'interpretation': interpretation,
        'consistency_level': consistency_level,
        'balance_assessment': balance_assessment,
        'auc_std': auc_std,
        'auc_diff': auc_diff,
        'best_classes': best_classes[:3],
        'worst_classes': best_classes[-3:],
        'poor_classes': poor_classes,
        'metrics': {
            'micro_auc': roc_auc_micro,
            'macro_auc': roc_auc_macro,
            'mean_auc': np.mean(list(roc_auc.values())),
            'std_auc': auc_std
        }
    }



def create_roc_curves_multiclass(model, test_loader, device, n_classes=10, figsize=(15, 5)):
    """
    Create ROC curves for multi-class classification using One-vs-Rest approach.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run model on
        n_classes: Number of classes (default: 10)
        figsize: Figure size for the plot
    
    Returns:
        dict: Dictionary containing ROC data and AUC scores
    """    
    model.eval()
    
    # Collect all test data and predictions
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions and probabilities
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Store results
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)
    
    # Binarize the output labels for One-vs-Rest ROC
    targets_bin = label_binarize(targets, classes=range(n_classes))
    
    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(targets_bin.ravel(), probabilities.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    # Calculate macro-average ROC curve and ROC area
    fpr_macro = np.linspace(0, 1, 100)
    tpr_macro = np.zeros_like(fpr_macro)
    for i in range(n_classes):
        tpr_macro += np.interp(fpr_macro, fpr[i], tpr[i])
    tpr_macro /= n_classes
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    
    # Plot ROC curves
    plt.figure(figsize=figsize)
    
    # Individual ROC curves for each class
    plt.subplot(1, 3, 1)
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, 
                 label=f'Digit {i} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Micro-average ROC curve
    plt.subplot(1, 3, 2)
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Average ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Macro-average ROC curve
    plt.subplot(1, 3, 3)
    plt.plot(fpr_macro, tpr_macro, color='navy', linestyle=':', linewidth=4,
             label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Average ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print basic results
    print("ROC Analysis Results:")
    print(f"  Micro-average AUC: {roc_auc_micro:.3f}")
    print(f"  Macro-average AUC: {roc_auc_macro:.3f}")
    print(f"\nPer-class AUC scores:")
    for i in range(n_classes):
        print(f"  Digit {i}: {roc_auc[i]:.3f}")
    
    print(f"\nROC curve plotting completed!")
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'fpr_micro': fpr_micro,
        'tpr_micro': tpr_micro,
        'roc_auc_micro': roc_auc_micro,
        'fpr_macro': fpr_macro,
        'tpr_macro': tpr_macro,
        'roc_auc_macro': roc_auc_macro,
        'targets': targets,
        'probabilities': probabilities
    }




def print_essential_summary(
    train_loader, val_loader, test_loader,
    model, 
    train_losses, val_losses, train_accuracies, val_accuracies,
    test_loss, test_acc, all_preds, all_actuals,
    roc_results=None,
    n_classes=10
):
    """
    Print essential summary statistics for the MNIST classification experiment.
    """
    
    print("="*60)
    print("MNIST CLASSIFICATION - EXPERIMENT SUMMARY")
    print("="*60)
    
    # Dataset Information
    print(f"Dataset Information:")
    print(f"  - Total samples: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")
    print(f"  - Training: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print(f"  - Classes: {n_classes} (digits 0-9)")
    print(f"  - Input size: 28×28 pixels")
    
    # Model Information
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Information:")
    print(f"  - Architecture: 3-layer MLP")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Optimizer: Adam")
    print(f"  - Loss function: CrossEntropyLoss")
    
    # Training Results
    epochs = len(train_losses)
    best_val_acc = max(val_accuracies)
    print(f"\nTraining Results:")
    print(f"  - Epochs trained: {epochs}")
    print(f"  - Final training accuracy: {train_accuracies[-1]:.2f}%")
    print(f"  - Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  - Final validation accuracy: {val_accuracies[-1]:.2f}%")
    
    # Test Performance
    correct_preds = sum(np.array(all_preds) == np.array(all_actuals))
    print(f"\nTest Performance:")
    print(f"  - Test accuracy: {test_acc:.2f}%")
    print(f"  - Test loss: {test_loss:.4f}")
    print(f"  - Correct predictions: {correct_preds}/{len(all_preds)}")
    
    # Per-class Performance
    cm = confusion_matrix(all_actuals, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\nPer-Class Accuracy:")
    for i in range(n_classes):
        print(f"  - Digit {i}: {per_class_acc[i]*100:.1f}%")
    
    print(f"\nClass Performance Summary:")
    print(f"  - Mean accuracy: {np.mean(per_class_acc)*100:.2f}%")
    print(f"  - Best class: Digit {np.argmax(per_class_acc)} ({max(per_class_acc)*100:.1f}%)")
    print(f"  - Worst class: Digit {np.argmin(per_class_acc)} ({min(per_class_acc)*100:.1f}%)")
    print(f"  - Standard deviation: {np.std(per_class_acc)*100:.2f}%")
    
    # ROC Performance (if available)
    if roc_results is not None:
        print(f"\nROC Analysis:")
        print(f"  - Micro-average AUC: {roc_results['roc_auc_micro']:.3f}")
        print(f"  - Macro-average AUC: {roc_results['roc_auc_macro']:.3f}")
    
    # Overfitting Check
    train_val_gap = train_accuracies[-1] - val_accuracies[-1]
    print(f"\nModel Assessment:")
    print(f"  - Train-Validation gap: {train_val_gap:.2f}%")
    
    if train_val_gap > 15:
        overfitting_status = "High overfitting detected"
    elif train_val_gap > 10:
        overfitting_status = "Mild overfitting"
    elif train_val_gap > 5:
        overfitting_status = "Good generalization"
    else:
        overfitting_status = "Excellent generalization"
    
    print(f"  - Status: {overfitting_status}")
    
    # Overall Performance Rating
    if test_acc >= 90:
        rating = "Excellent"
    elif test_acc >= 80:
        rating = "Good"
    elif test_acc >= 70:
        rating = "Fair"
    else:
        rating = "Needs Improvement"
    
    print(f"  - Overall rating: {rating}")
    
    print("="*60)



def plot_per_class_accuracy(targets, predictions, n_classes=10, figure_size=(10, 6), 
                           bar_color='skyblue', bar_edge_color='navy', grid_alpha=0.3, 
                           y_limit=(0, 105), title='Per-Class Accuracy [%]'):
    """
    Plot per-class accuracy histogram.
    
    Args:
        targets: True labels
        predictions: Predicted labels
        n_classes: Number of classes (default: 10)
        figure_size: Figure size (default: (10, 6))
        bar_color: Color for accuracy bars (default: 'skyblue')
        bar_edge_color: Edge color for accuracy bars (default: 'navy')
        grid_alpha: Grid transparency (default: 0.3)
        y_limit: Y-axis limits (default: (0, 105))
        title: Plot title (default: 'Per-Class Accuracy [%]')
    """
    
    # Create confusion matrix to calculate per-class accuracy
    cm = confusion_matrix(targets, predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Plot per-class accuracy
    plt.figure(figsize=figure_size)
    bars = plt.bar(range(n_classes), per_class_acc * 100, color=bar_color, edgecolor=bar_edge_color)
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.xticks(range(n_classes))
    plt.ylim(y_limit)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.grid(True, alpha=grid_alpha)
    plt.show()
    
    # Print per-class accuracy summary
    print("Per-Class Accuracy Summary:")
    for i in range(n_classes):
        print(f"  Class {i}: {per_class_acc[i]*100:.2f}%")
    
    print(f"  Mean accuracy: {np.mean(per_class_acc)*100:.2f}%")
    print(f"  Std accuracy: {np.std(per_class_acc)*100:.2f}%")

