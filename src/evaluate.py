# Description: This file contains functions to evaluate the model and plot loss and accuracy curves.
from sklearn.metrics import classification_report
from typing import Dict, List
import matplotlib.pyplot as plt
import torch
from config import device

def plot_loss_and_accuracy_curves(results: Dict[str, List[float]]):
    """Plots training and testing loss and accuracy curves for Age Classification."""
    
    # Extract losses
    train_loss = results['train_loss']
    test_loss = results['test_loss']

    # Extract Age Accuracy
    train_acc = results['train_acc']
    test_acc = results['test_acc']

    epochs = range(len(train_loss))

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Age Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Age Accuracy')
    plt.plot(epochs, test_acc, label='Test Age Accuracy')
    plt.title('Age Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('age_classification_results.png')
    plt.show()


def predict(model, dataloader, device=device):
    """
    Perform prediction for Age Classification Only.

    Parameters:
        model: PyTorch CNN model (only for age classification).
        dataloader: DataLoader for test data.
        device: Device for computation (e.g., "cpu" or "cuda").

    Returns:
        results: Dictionary containing predicted and true age classes.
    """
    age_pred_classes, age_true_classes = [], []

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            age_pred = model(X)

            # Get predicted class
            age_pred_class = age_pred.argmax(dim=1).tolist()
            age_true_class = y.tolist()
            
            # Store results
            age_pred_classes.extend(age_pred_class)
            age_true_classes.extend(age_true_class)

    # Return results
    results = {
        "Age": {'pred': age_pred_classes, 'true': age_true_classes}
    }
    
    return results


def evaluate_predictions(eval_results, age_idx):
    """
    Evaluates model predictions and prints a classification report.
    
    Parameters:
    eval_results (dict): The evaluation results containing 'Age' with 'true' and 'pred' values.
    age_idx (dict): Dictionary mapping age categories to indices.
    
    Returns:
    str: The classification report as a string.
    """
    # Extract valid indices (if -1 is used as a missing value indicator)
    valid_age_indices = [i for i, value in enumerate(eval_results['Age']['true']) if value != -1]
    
    # Filter out valid predictions and true labels
    valid_true_ages = [eval_results['Age']['true'][i] for i in valid_age_indices]
    valid_pred_ages = [eval_results['Age']['pred'][i] for i in valid_age_indices]
    
    # Ensure target names match age bins
    age_names = list(age_idx.keys())  # Extract names from age_idx dictionary
    
    # Generate classification report
    report = classification_report(y_true=valid_true_ages, y_pred=valid_pred_ages, target_names=age_names)
    
    # print(report)
    return report
