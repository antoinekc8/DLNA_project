import torch
import numpy as np
from sklearn.metrics import mean_squared_error

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
