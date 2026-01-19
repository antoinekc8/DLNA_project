import os
import random 
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.model_selection import train_test_split


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



def compare_architectures(train_loader, val_loader, test_loader, device, 
                         model_configs, epochs=50, task_type='classification'):
    """
    Generic model comparison function that can compare any types of models
    
    Args:
        train_loader, val_loader, test_loader: Data loaders
        device: Computing device  
        model_configs: List of dictionaries, each containing:
            - "name": str - Name for this configuration
            - "model": nn.Module - The actual model instance (already initialized)
            - "optimizer": torch.optim.Optimizer - The optimizer instance
            - "loss_fn": torch.nn loss function - Loss function to use
        epochs: Number of training epochs
        task_type: str - "classification" or "regression"
    
    Returns:
        results: Dict with test accuracies and metrics
        training_histories: Dict with training curves
    """
    
    results = {}
    training_histories = {}

    print(f"Starting {task_type} model comparison...")
    print("=" * 60)
    
    for i, config in enumerate(model_configs):
        print(f"\n{i+1}/{len(model_configs)}: Testing {config['name']}")
        print("-" * 40)
        
        # Get model, optimizer, and loss function from config
        model = config["model"]
        optimizer = config["optimizer"] 
        loss_fn = config["loss_fn"]
        
        # Train with specified task type
        if task_type == 'regression':
            trained_model, train_losses, val_losses, train_metrics, val_metrics = train_with_validation(
                model, train_loader, val_loader, optimizer, loss_fn, device, 
                epochs=epochs, task_type='regression'
            )
        else:  # classification
            trained_model, train_losses, val_losses, train_metrics, val_metrics = train_with_validation(
                model, train_loader, val_loader, optimizer, loss_fn, device, 
                epochs=epochs, task_type='classification'
            )
        
        # Test evaluation with specified task type
        if task_type == 'regression':
            test_loss, test_metric, predictions, actuals = test_model(
                trained_model, test_loader, loss_fn, device, task_type='regression'
            )
            metric_name = 'mse'
        else:  # classification
            test_loss, test_metric, predictions, actuals = test_model(
                trained_model, test_loader, loss_fn, device, task_type='classification'
            )
            metric_name = 'test_accuracy'

        # Store results (unified format for both tasks)
        results[config["name"]] = {
            'model': trained_model,
            'test_loss': test_loss,
            metric_name: test_metric,
            'final_val_metric': val_metrics[-1] if val_metrics else 0,
            'parameters': sum(p.numel() for p in trained_model.parameters()),
            'predictions': predictions,
            'actuals': actuals
        }
        
        training_histories[config["name"]] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        if task_type == 'regression':
            print(f"Test Loss: {test_loss:.6f} | RMSE: {test_metric:.6f} | Parameters: {results[config['name']]['parameters']:,}")
        else:
            print(f"Test Accuracy: {test_metric:.2f}% | Parameters: {results[config['name']]['parameters']:,}")
    
    # Display results summary
    print("\n" + "=" * 70)
    print(f"{task_type.upper()} MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    if task_type == 'regression':
        print(f"{'Model':<25} {'Test Loss':<12} {'MSE':<12} {'Parameters':<12}")
        print("-" * 70)
        for name, metrics in results.items():
            print(f"{name:<25} {metrics['test_loss']:<12.6f} {metrics['mse']:<12.6f} {metrics['parameters']:<12,}")
    else:
        print(f"{'Model':<25} {'Test Acc':<10} {'Val Acc':<10} {'Parameters':<12}")
        print("-" * 70)
        for name, metrics in results.items():
            print(f"{name:<25} {metrics['test_accuracy']:<10.2f} "
                  f"{metrics['final_val_metric']:<10.2f} {metrics['parameters']:<12,}")
    
    return results, training_histories




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



def plot_architecture_comparison(results, histories=None, task_type='classification'):
    """
    Unified plotting function for both classification and regression model comparison
    
    Args:
        results: Results dictionary from compare_architectures
        histories: Training histories (optional)
        task_type: 'classification' or 'regression'
    """
    # Extract data for plotting
    names = list(results.keys())
    parameters = [results[name]['parameters'] for name in names]
    
    if task_type == 'classification':
        primary_metric = [results[name]['test_accuracy'] for name in names]
        val_metric = [results[name]['final_val_metric'] for name in names]
        metric_name = 'Accuracy (%)'
        primary_label = 'Test Accuracy'
        val_label = 'Validation Accuracy'
        format_str = '{:.1f}%'
    else:  # regression
        primary_metric = [results[name]['mse'] for name in names]
        val_metric = [results[name]['final_val_metric'] for name in names]  # val RMSE
        metric_name = 'MSE'
        primary_label = 'Test MSE'
        val_label = 'Validation MSE'
        format_str = '{:.4f}'
    
    # Determine subplot layout based on available data
    if histories:
        # Full 2x2 layout with training curves
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17, 15))
        axes = [ax1, ax2, ax3, ax4]
    else:
        # 1x3 layout without training curves
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        axes = [ax1, ax2, ax3]
    
    # Plot 1: Primary metric comparison
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, primary_metric, width, 
                   label=primary_label, color='skyblue', edgecolor='navy')
    bars2 = ax1.bar(x_pos + width/2, val_metric, width,
                   label=val_label, color='lightcoral', edgecolor='darkred')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            offset = 0.5 if task_type == 'classification' else height * 0.02
            ax1.text(bar.get_x() + bar.get_width()/2., height + offset,
                    format_str.format(height), ha='center', va='bottom', fontsize=9)
    
    ax1.set_title(f'{primary_label} vs {val_label}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel(metric_name)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter count vs Performance
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    scatter = ax2.scatter(parameters, primary_metric, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    for i, name in enumerate(names):
        ax2.annotate(name, (parameters[i], primary_metric[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_title(f'Parameter Count vs {primary_label}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel(metric_name)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model efficiency (Performance per parameter)
    if task_type == 'classification':
        efficiency = [acc/params*1000 for acc, params in zip(primary_metric, parameters)]
        efficiency_label = 'Accuracy per 1K Parameters'
    else:
        # For regression, lower MSE is better, so we use inverse
        efficiency = [1000/(mse * params) for mse, params in zip(primary_metric, parameters)]
        efficiency_label = 'Efficiency (1000/(MSE × Params))'
    
    bars = ax3.bar(names, efficiency, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{eff:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_title(f'Model Efficiency', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel(efficiency_label)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training curves (only if histories provided and using 2x2 layout)
    if histories and len(axes) == 4:
        ax4 = axes[3]
        for i, (name, history) in enumerate(histories.items()):
            color = colors[i]
            epochs = range(1, len(history['train_losses']) + 1)
            ax4.plot(epochs, history['train_losses'], '--', color=color, alpha=0.7, label=f'{name} Train')
            ax4.plot(epochs, history['val_losses'], '-', color=color, label=f'{name} Val')
        
        ax4.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{task_type.upper()} COMPARISON SUMMARY:")
    print("="*50)
    
    if task_type == 'classification':
        best_idx = np.argmax(primary_metric)
        print(f"Best performing: {names[best_idx]} ({primary_metric[best_idx]:.2f}%)")
    else:
        best_idx = np.argmin(primary_metric)  # Lower MSE is better
        print(f"Best performing: {names[best_idx]} (MSE: {primary_metric[best_idx]:.4f})")
    
    most_efficient_idx = np.argmax(efficiency)
    print(f"Most efficient: {names[most_efficient_idx]}")
    print(f"Parameter range: {min(parameters):,} - {max(parameters):,}")



def plot_architecture_comparison(results, histories=None, task_type='classification'):
    """
    Unified plotting function for both classification and regression model comparison
    
    Args:
        results: Results dictionary from compare_architectures
        histories: Training histories (optional)
        task_type: 'classification' or 'regression'
    """
    # Extract data for plotting
    names = list(results.keys())
    parameters = [results[name]['parameters'] for name in names]
    
    if task_type == 'classification':
        primary_metric = [results[name]['test_accuracy'] for name in names]
        val_metric = [results[name]['final_val_metric'] for name in names]
        metric_name = 'Accuracy (%)'
        primary_label = 'Test Accuracy'
        val_label = 'Validation Accuracy'
        format_str = '{:.1f}%'
    else:  # regression
        primary_metric = [results[name]['mse'] for name in names]
        val_metric = [results[name]['final_val_metric'] for name in names]  # val RMSE
        metric_name = 'MSE'
        primary_label = 'Test MSE'
        val_label = 'Validation MSE'
        format_str = '{:.4f}'
    
    # Determine subplot layout based on available data
    if histories:
        # Full 2x2 layout with training curves
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17, 15))
        axes = [ax1, ax2, ax3, ax4]
    else:
        # 1x3 layout without training curves
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        axes = [ax1, ax2, ax3]
    
    # Plot 1: Primary metric comparison
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, primary_metric, width, 
                   label=primary_label, color='skyblue', edgecolor='navy')
    bars2 = ax1.bar(x_pos + width/2, val_metric, width,
                   label=val_label, color='lightcoral', edgecolor='darkred')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            offset = 0.5 if task_type == 'classification' else height * 0.02
            ax1.text(bar.get_x() + bar.get_width()/2., height + offset,
                    format_str.format(height), ha='center', va='bottom', fontsize=9)
    
    ax1.set_title(f'{primary_label} vs {val_label}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel(metric_name)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter count vs Performance
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    scatter = ax2.scatter(parameters, primary_metric, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    for i, name in enumerate(names):
        ax2.annotate(name, (parameters[i], primary_metric[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_title(f'Parameter Count vs {primary_label}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel(metric_name)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model efficiency (Performance per parameter)
    if task_type == 'classification':
        efficiency = [acc/params*1000 for acc, params in zip(primary_metric, parameters)]
        efficiency_label = 'Accuracy per 1K Parameters'
    else:
        # For regression, lower MSE is better, so we use inverse
        efficiency = [1000/(mse * params) for mse, params in zip(primary_metric, parameters)]
        efficiency_label = 'Efficiency (1000/(MSE × Params))'
    
    bars = ax3.bar(names, efficiency, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{eff:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_title(f'Model Efficiency', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel(efficiency_label)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training curves (only if histories provided and using 2x2 layout)
    if histories and len(axes) == 4:
        ax4 = axes[3]
        for i, (name, history) in enumerate(histories.items()):
            color = colors[i]
            epochs = range(1, len(history['train_losses']) + 1)
            ax4.plot(epochs, history['train_losses'], '--', color=color, alpha=0.7, label=f'{name} Train')
            ax4.plot(epochs, history['val_losses'], '-', color=color, label=f'{name} Val')
        
        ax4.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{task_type.upper()} COMPARISON SUMMARY:")
    print("="*50)
    
    if task_type == 'classification':
        best_idx = np.argmax(primary_metric)
        print(f"Best performing: {names[best_idx]} ({primary_metric[best_idx]:.2f}%)")
    else:
        best_idx = np.argmin(primary_metric)  # Lower MSE is better
        print(f"Best performing: {names[best_idx]} (MSE: {primary_metric[best_idx]:.4f})")
    
    most_efficient_idx = np.argmax(efficiency)
    print(f"Most efficient: {names[most_efficient_idx]}")
    print(f"Parameter range: {min(parameters):,} - {max(parameters):,}")


def train_progressive_network(model, train_loader, val_loader, test_loader, device,
                            initial_epochs=10, growth_epochs=5, max_depth=4):

    """
    Train network with progressive growth using ProgressiveNet
    
    Parameters:
    -----------
    model : ProgressiveNet
        The progressive network model that can grow during training
    train_loader : DataLoader
        Training data loader containing batches of (images, labels)
    val_loader : DataLoader
        Validation data loader for monitoring training progress
    test_loader : DataLoader
        Test data loader for final performance evaluation
    device : torch.device
        Device to run training on ('cuda' or 'cpu')
    initial_epochs : int, default=10
        Number of epochs to train the initial (depth=1) network
    growth_epochs : int, default=5
        Number of epochs to train each newly added layer
    max_depth : int, default=4
        Maximum depth (number of layers) the network can grow to
        
    Returns:
    --------
    model : ProgressiveNet
        The trained progressive network at maximum depth
    results : dict
        Dictionary containing training history and performance metrics
    """
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters()) # use the same optimizer as in Part 2
    criterion = nn.CrossEntropyLoss() # use the same loss function as in Part 2
    
    results = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'architectures': [],
        'depth_performance': {}  # Track performance at each depth
    }
    
    # Train at each depth level
    while model.current_depth <= max_depth:
        print(f"\nTraining Depth {model.current_depth}")
        print("=" * 50)
        # inspect_architecture(model, f"Depth {model.current_depth}")
        results['architectures'].append(f"Depth {model.current_depth}")
        
        # Determine epochs for this depth
        epochs = initial_epochs if model.current_depth == 1 else growth_epochs
        
        # Train model at current depth
        model, train_losses, val_losses, train_accs, val_accs = train_with_validation(
            model, train_loader, val_loader, optimizer, criterion, device, 
            epochs=epochs
        )
        
        results['train_losses'].extend(train_losses)
        results['val_losses'].extend(val_losses)
        results['train_accs'].extend(train_accs)
        results['val_accs'].extend(val_accs)
        
        # Test model at current depth
        test_loss, test_acc, test_preds, test_actuals = test_model(
            model, test_loader, criterion, device
        )
        results['depth_performance'][model.current_depth] = {
            'test_loss': test_loss, 
            'test_acc': test_acc,
            'test_preds': test_preds,
            'test_actuals': test_actuals
        }
        print(f"Depth {model.current_depth} Test Accuracy: {test_acc:.2f}%")
        
        # Check if we should add another block
        if model.current_depth < max_depth:
            model._add_block()
            model = model.to(device)  # Move updated model to device
            # Update optimizer for new parameters
            optimizer = optim.Adam(model.parameters())
        else:
            break
    
    # Final comparison across all depths
    print(f"\nPerformance Comparison Across Depths:")
    print("=" * 50)
    for depth, perf in results['depth_performance'].items():
        print(f"Depth {depth}: {perf['test_acc']:.2f}% accuracy, {perf['test_loss']:.4f} loss")
    
    return model, results




def analyze_progressive_results(results):
    """Analyze and visualize progressive training results"""
    
    # Plot training curves with accuracies
    plo_training_results_with_accuracy(
        results['train_losses'], 
        results['val_losses'],
        results['train_accs'], 
        results['val_accs']
    )
    
    # Check if depth_performance exists
    if 'depth_performance' not in results or not results['depth_performance']:
        print(" No depth performance data found!")
        return
    
    # Create depth comparison plot
    depths = list(results['depth_performance'].keys())
    accuracies = [results['depth_performance'][d]['test_acc'] for d in depths]
    losses = [results['depth_performance'][d]['test_loss'] for d in depths]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy vs Depth
    ax1.plot(depths, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Network Depth')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy vs Network Depth')
    ax1.grid(True, alpha=0.3)
    for i, acc in enumerate(accuracies):
        ax1.annotate(f'{acc:.1f}%', (depths[i], acc), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Loss vs Depth  
    ax2.plot(depths, losses, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Network Depth')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss vs Network Depth')
    ax2.grid(True, alpha=0.3)
    for i, loss in enumerate(losses):
        ax2.annotate(f'{loss:.3f}', (depths[i], loss), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal depth
    best_depth = max(results['depth_performance'].keys(), 
                    key=lambda d: results['depth_performance'][d]['test_acc'])
    best_acc = results['depth_performance'][best_depth]['test_acc']
    
    print(f"\n Analysis Summary:")
    print("=" * 50)
    print(f"Best performing depth: {best_depth}")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Performance improvement: {accuracies[-1] - accuracies[0]:.2f}%")


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



def build_sequences(data, id_col="station_id", time_col="timestamp", 
                   target_col="available_bikes", seq_len=24):
    """
    Build time series sequences from raw data for training
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw time series data with station_id, timestamp, and target columns
    id_col : str, default="station_id"
        Column name for station identifier
    time_col : str, default="timestamp" 
        Column name for timestamp
    target_col : str, default="available_bikes"
        Column name for target variable
    seq_len : int, default=24
        Length of input sequences (time steps)
        
    Returns:
    --------
    tuple : (all_sequences, all_targets)
        all_sequences: List of input sequences (N, seq_len, 1)
        all_targets: List of target values (N, 1)
    """
    if id_col not in data or time_col not in data or target_col not in data:
        missing = [c for c in [id_col, time_col, target_col] if c not in data]
        raise ValueError(f"Missing columns: {missing}")

    df = data.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    stations = df[id_col].unique()
    print(f"{len(stations)} stations -> {sorted(stations)[:8]}{' ...' if len(stations)>8 else ''}")
    print(f"Time span: {df[time_col].min()} to {df[time_col].max()}")

    all_sequences, all_targets = [], []
    for sid in stations:
        s = df[df[id_col] == sid]
        v = s[[target_col]].to_numpy()  # (T,1)
        if len(v) < seq_len + 1:
            print(f"skip {sid}: {len(v)} rows < {seq_len+1}")
            continue
        for i in range(len(v) - seq_len):
            all_sequences.append(v[i:i+seq_len])   # (seq_len,1)
            all_targets.append(v[i+seq_len])       # (1,)

    if not all_sequences:
        raise ValueError("No valid sequences created. Check data length and seq_len.")
    return all_sequences, all_targets



def scale_split_loaders(all_sequences, all_targets, seq_len=24, 
                       train_ratio=0.7, val_ratio=0.15, batch_size=32, 
                       random_state=42, scaler=None):
    """
    Scale time series data and split into train/validation/test DataLoaders
    
    Parameters:
    -----------
    all_sequences : list or np.ndarray
        Input sequences of shape (N, seq_len, 1)
    all_targets : list or np.ndarray  
        Target values of shape (N, 1)
    seq_len : int, default=24
        Length of input sequences
    train_ratio : float, default=0.7
        Proportion of data for training (0 < train_ratio < 1)
    val_ratio : float, default=0.15
        Proportion of data for validation (0 < val_ratio < 1)
    batch_size : int, default=32
        Batch size for DataLoaders
    random_state : int, default=42
        Random seed for reproducible splits
    scaler : MinMaxScaler, optional
        Pre-fitted scaler. If None, will fit new scaler
        
    Returns:
    --------
    tuple : (train_loader, val_loader, test_loader, scaler)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data  
        test_loader: DataLoader for test data
        scaler: Fitted MinMaxScaler for inverse transformation
        
    Raises:
    -------
    ValueError: If ratios are invalid or data is empty
    """
    # Validate input data
    if not all_sequences or not all_targets:
        raise ValueError("Sequences and targets cannot be empty")
    
    if len(all_sequences) != len(all_targets):
        raise ValueError("Number of sequences must match number of targets")
    
    # Convert to numpy arrays
    X = np.array(all_sequences)  # (N, seq_len, 1)
    y = np.array(all_targets)    # (N, 1)
    print(f"Sequences: {X.shape}, Targets: {y.shape}")

    # Validate ratios
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be between 0 and 1")
    
    test_ratio = 1 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    # Scale data using MinMaxScaler
    if scaler is None:
        scaler = MinMaxScaler()
        # Fit scaler on flattened sequences
        X_flat = X.reshape(-1, 1)  # (N*seq_len, 1)
        scaler.fit(X_flat)
    
    # Transform data
    X_flat = X.reshape(-1, 1)
    X_scaled = scaler.transform(X_flat).reshape(X.shape)
    y_scaled = scaler.transform(y)

    # Split data into train/validation/test sets
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_ratio, random_state=random_state
    )
    
    # Calculate validation size relative to remaining data
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, random_state=random_state
    )

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), 
        batch_size=batch_size, 
        shuffle=False
    )

    # Print split summary
    print(f"Data split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train ratio: {len(X_train)/len(X):.3f}, Val ratio: {len(X_val)/len(X):.3f}, Test ratio: {len(X_test)/len(X):.3f}")
    
    return train_loader, val_loader, test_loader, scaler


def create_velov_model_configs(models_dict, device, learning_rate=0.001, 
                              optimizer_class=optim.Adam, loss_fn=nn.MSELoss(),
                              optimizer_kwargs=None):
    """
    Create model configurations for Velov time series comparison using existing models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary containing model instances {'LSTM': model, 'GRU': model, ...}
    device : torch.device
        Device to run models on ('cuda' or 'cpu')
    learning_rate : float, default=0.001
        Learning rate for optimizers
    optimizer_class : torch.optim.Optimizer, default=optim.Adam
        Optimizer class to use for all models
    loss_fn : torch.nn.Module, default=nn.MSELoss()
        Loss function to use for all models
    optimizer_kwargs : dict, optional
        Additional keyword arguments for optimizer (e.g., weight_decay, betas)
        
    Returns:
    --------
    list : List of model configuration dictionaries
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    model_configs = []
    
    for name, model in models_dict.items():
        # Move model to device
        model = model.to(device)
        
        # Create optimizer with learning rate and additional kwargs
        optimizer = optimizer_class(model.parameters(), lr=learning_rate, **optimizer_kwargs)
        
        # Create configuration
        config = {
            "name": name,
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn
        }
        model_configs.append(config)
    
    return model_configs




def plot_all_time_series_results(results):
    """
    Plot time series regression results for all models in a 2x2 grid
    Each subplot shows predictions vs actuals for one model
    
    Args:
        results: Dictionary containing results for all models
                Each model should have 'actuals', 'predictions', and 'mse'
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Make it easier to iterate
    
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Predictions vs Actuals scatter plot
        ax.scatter(result['actuals'], result['predictions'], alpha=0.6, s=20)
        ax.plot([min(result['actuals']), max(result['actuals'])], 
                [min(result['actuals']), max(result['actuals'])], 'r--', alpha=0.8)
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{name} - MSE: {result["mse"]:.4f}')
        ax.grid(True, alpha=0.3)
        
        # Add R² score if possible
        try:
            import numpy as np
            actuals = np.array(result['actuals'])
            predictions = np.array(result['predictions'])
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
    
    plt.tight_layout()
    plt.show()


def create_architecture_search_configs(model_classes, search_space, device, learning_rate=0.001, 
                                     num_configs=100, optimizer_class=optim.Adam, 
                                     loss_fn=nn.MSELoss(), optimizer_kwargs=None):
    """
    Generate diverse architecture configurations for automated search using model classes
    
    Parameters:
    -----------
    model_classes : dict
        Dictionary containing model classes {'LSTM': VelovLSTM, 'GRU': VelovGRU, ...}
    search_space : dict
        Dictionary defining search space with keys: model_type, hidden_size, num_layers, dropout
    device : torch.device
        Device to run models on ('cuda' or 'cpu')
    learning_rate : float, default=0.001
        Learning rate for optimizers
    num_configs : int, default=100
        Maximum number of configurations to generate
    optimizer_class : torch.optim.Optimizer, default=optim.Adam
        Optimizer class to use for all models
    loss_fn : torch.nn.Module, default=nn.MSELoss()
        Loss function to use for all models
    optimizer_kwargs : dict, optional
        Additional keyword arguments for optimizer
        
    Returns:
    --------
    list : List of model configuration dictionaries
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    import itertools, random    
    configs = []
    
    # Generate all possible combinations (limited to avoid explosion)
    all_combinations = list(itertools.product(
        search_space['model_type'],
        search_space['hidden_size'], 
        search_space['num_layers'],
        search_space['dropout']
    ))
    
    # Randomly sample configurations if too many
    if len(all_combinations) > num_configs:
        selected_combinations = random.sample(all_combinations, num_configs)
    else:
        selected_combinations = all_combinations
    
    print(f"Testing {len(selected_combinations)} architecture configurations...")
    
    for i, (model_type, hidden_size, num_layers, dropout) in enumerate(selected_combinations):
        # Create model instance using the model class
        if model_type in model_classes:
            if model_type == 'MLP':
                model = model_classes[model_type](
                    sequence_length=24,
                    hidden_size=hidden_size, 
                    num_layers=num_layers, 
                    dropout=dropout
                ).to(device)
            else:
                model = model_classes[model_type](
                    input_size=1, 
                    hidden_size=hidden_size, 
                    num_layers=num_layers, 
                    dropout=dropout
                ).to(device)
        else:
            print(f"Warning: Unknown model type {model_type}, skipping...")
            continue
        
        # Create optimizer
        optimizer = optimizer_class(model.parameters(), lr=learning_rate, **optimizer_kwargs)
        
        # Create configuration
        config_name = f"{model_type}_h{hidden_size}_l{num_layers}_d{dropout}"
        config = {
            "name": config_name,
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "params": {
                'model_type': model_type,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout
            }
        }
        configs.append(config)
    
    return configs




# Plot Architecture Search Results
def plot_architecture_search_results(results, search_configs):
    """
    Visualize architecture search results with detailed analysis
    """
    
    # Extract data for analysis
    names = list(results.keys())
    mse_scores = [results[name]['mse'] for name in names]
    param_counts = [results[name]['parameters'] for name in names]
    
    # Extract architecture parameters
    model_types = []
    hidden_sizes = []
    num_layers = []
    dropouts = []
    
    for config in search_configs:
        params = config['params']
        model_types.append(params['model_type'])
        hidden_sizes.append(params['hidden_size'])
        num_layers.append(params['num_layers'])
        dropouts.append(params['dropout'])
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. MSE vs Model Type
    model_type_mse = {}
    for i, mtype in enumerate(model_types):
        if mtype not in model_type_mse:
            model_type_mse[mtype] = []
        model_type_mse[mtype].append(mse_scores[i])
    
    types = list(model_type_mse.keys())
    avg_mse = [np.mean(model_type_mse[t]) for t in types]
    std_mse = [np.std(model_type_mse[t]) for t in types]
    
    ax1.bar(types, avg_mse, yerr=std_mse, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Average MSE by Model Type', fontweight='bold')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (avg, std) in enumerate(zip(avg_mse, std_mse)):
        ax1.text(i, avg + std + 0.001, f'{avg:.4f}', ha='center', fontweight='bold')
    
    # 2. MSE vs Hidden Size
    hidden_size_mse = {}
    for i, hsize in enumerate(hidden_sizes):
        if hsize not in hidden_size_mse:
            hidden_size_mse[hsize] = []
        hidden_size_mse[hsize].append(mse_scores[i])
    
    sizes = sorted(hidden_size_mse.keys())
    avg_mse_hidden = [np.mean(hidden_size_mse[s]) for s in sizes]
    
    ax2.plot(sizes, avg_mse_hidden, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_title('MSE vs Hidden Size', fontweight='bold')
    ax2.set_xlabel('Hidden Size')
    ax2.set_ylabel('Average MSE')
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameter Count vs Performance
    colors = ['red' if t == 'LSTM' else 'blue' if t == 'GRU' else 'green' for t in model_types]
    scatter = ax3.scatter(param_counts, mse_scores, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add best model annotation
    best_idx = np.argmin(mse_scores)
    ax3.annotate(f'Best: {names[best_idx]}', 
                xy=(param_counts[best_idx], mse_scores[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax3.set_title('Parameter Count vs MSE', fontweight='bold')
    ax3.set_xlabel('Number of Parameters')
    ax3.set_ylabel('MSE')
    ax3.grid(True, alpha=0.3)
    
    # Add legend for model types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='LSTM'),
                      Patch(facecolor='blue', label='GRU'),
                      Patch(facecolor='green', label='RNN')]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # 4. Top 5 Configurations
    sorted_indices = np.argsort(mse_scores)[:5]
    top_names = [names[i] for i in sorted_indices]
    top_scores = [mse_scores[i] for i in sorted_indices]
    
    bars = ax4.barh(range(len(top_names)), top_scores, color='lightblue', alpha=0.8)
    ax4.set_yticks(range(len(top_names)))
    ax4.set_yticklabels([name.replace('_', '\n') for name in top_names], fontsize=8)
    ax4.set_title('Top 5 Configurations', fontweight='bold')
    ax4.set_xlabel('MSE')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax4.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("ARCHITECTURE SEARCH ANALYSIS")
    print("="*60)
    
    best_idx = np.argmin(mse_scores)
    best_config = search_configs[best_idx]
    
    print(f"\nBest Configuration:")
    print(f"  Name: {names[best_idx]}")
    print(f"  MSE: {mse_scores[best_idx]:.6f}")
    print(f"  Parameters: {param_counts[best_idx]:,}")
    print(f"  Architecture: {best_config['params']}")
    
    print(f"\nModel Type Performance:")
    for mtype in types:
        avg = np.mean(model_type_mse[mtype])
        print(f"  {mtype}: {avg:.6f} ± {np.std(model_type_mse[mtype]):.6f}")
    
    print(f"\nParameter Efficiency (MSE per 1K parameters):")
    efficiencies = [(mse_scores[i] * 1000) / param_counts[i] for i in range(len(names))]
    most_efficient_idx = np.argmin(efficiencies)
    print(f"  Most efficient: {names[most_efficient_idx]} ({efficiencies[most_efficient_idx]:.6f})")



