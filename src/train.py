# We will define the training and evaluation functions for the model in this file.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import device


# def train_step(model, dataloader, age_loss_fn, optimizer, device=device):
#     """
#     Perform a single training step for one epoch (Age Classification Only).

#     Parameters:
#         model: PyTorch CNN model (only for age classification).
#         dataloader: DataLoader for training data.
#         age_loss_fn: Loss function for age classification.
#         optimizer: Optimizer to update the model parameters.
#         device: Device for computation (e.g., "cpu" or "cuda").

#     Returns:
#         train_loss: Average loss over the epoch.
#         train_acc: Average age classification accuracy.
#     """
#     model.train()
#     train_loss = 0
#     total_age_accuracy = 0
#     num_batches = len(dataloader)

#     for X, y in dataloader:
#         X, y = X.to(device), y.long().to(device)

#         # Forward Pass
#         age_pred = model(X)

#         # Compute Age Loss
#         age_loss = age_loss_fn(age_pred, y)

#         # Compute Accuracy
#         age_pred_class = age_pred.argmax(dim=1)  # Get predicted classes
#         age_accuracy = (age_pred_class == y).sum().item() / y.size(0)
#         total_age_accuracy += age_accuracy

#         # Backpropagation
#         optimizer.zero_grad()
#         age_loss.backward()
#         optimizer.step()

#         # Accumulate Loss
#         train_loss += age_loss.item()

#     # Compute Average Loss and Accuracy
#     train_loss /= num_batches
#     avg_age_accuracy = total_age_accuracy / num_batches

#     return train_loss, avg_age_accuracy


# def test_step(model, dataloader, age_loss_fn, device=device):
#     """
#     Perform a single testing step for Age Classification Only.

#     Parameters:
#         model: PyTorch CNN model (only for age classification).
#         dataloader: DataLoader for test data.
#         age_loss_fn: Loss function for age classification.
#         device: Device for computation (e.g., "cpu" or "cuda").

#     Returns:
#         test_loss: Average loss over the epoch.
#         test_acc: Average accuracy for age classification.
#     """
#     model.eval()
#     test_loss = 0
#     total_age_accuracy = 0
#     num_batches = len(dataloader)

#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.long().to(device)  # Use only age labels

#             # Forward Pass
#             age_pred = model(X)

#             # Compute Age Loss
#             age_loss = age_loss_fn(age_pred, y)

#             # Compute Accuracy
#             age_pred_class = age_pred.argmax(dim=1)  # Get predicted classes
#             age_accuracy = (age_pred_class == y).sum().item() / y.size(0)
#             total_age_accuracy += age_accuracy

#             # Accumulate Total Loss
#             test_loss += age_loss.item()

#     # Compute Average Loss and Accuracy
#     test_loss /= num_batches
#     avg_age_accuracy = total_age_accuracy / num_batches

#     return test_loss, avg_age_accuracy



def train_step(model, dataloader, age_loss_fn, optimizer, device=device):
    """
    Perform a single training step for one epoch (Age Classification Only).
    """
    model.train()
    train_loss = 0
    total_age_accuracy = 0
    num_batches = len(dataloader)

    # Voeg tqdm voortgangsbalk toe
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for X, y in progress_bar:
        X, y = X.to(device), y.long().to(device)

        # Forward Pass
        age_pred = model(X)
        age_loss = age_loss_fn(age_pred, y)

        # Compute Accuracy
        age_pred_class = age_pred.argmax(dim=1)  # Get predicted classes
        age_accuracy = (age_pred_class == y).sum().item() / y.size(0)
        total_age_accuracy += age_accuracy

        # Backpropagation
        optimizer.zero_grad()
        age_loss.backward()
        optimizer.step()

        # Accumulate Loss
        train_loss += age_loss.item()

        # Update tqdm progress bar met loss en acc
        progress_bar.set_postfix(loss=age_loss.item(), acc=age_accuracy)

    # Compute Average Loss and Accuracy
    train_loss /= num_batches
    avg_age_accuracy = total_age_accuracy / num_batches

    return train_loss, avg_age_accuracy

def test_step(model, dataloader, age_loss_fn, device=device):
    """
    Perform a single testing step for Age Classification Only.
    """
    model.eval()
    test_loss = 0
    total_age_accuracy = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing", leave=False)

        for X, y in progress_bar:
            X, y = X.to(device), y.long().to(device)

            # Forward Pass
            age_pred = model(X)
            age_loss = age_loss_fn(age_pred, y)

            # Compute Accuracy
            age_pred_class = age_pred.argmax(dim=1)
            age_accuracy = (age_pred_class == y).sum().item() / y.size(0)
            total_age_accuracy += age_accuracy

            # Accumulate Total Loss
            test_loss += age_loss.item()

            # Update tqdm progress bar met loss en acc
            progress_bar.set_postfix(loss=age_loss.item(), acc=age_accuracy)

    # Compute Average Loss and Accuracy
    test_loss /= num_batches
    avg_age_accuracy = total_age_accuracy / num_batches

    return test_loss, avg_age_accuracy


def train(model, 
          train_dataloader, 
          test_dataloader, 
          optimizer, 
          age_loss_fn, 
          epochs, 
          device=device):
    """
    Training loop for Age Classification Only.

    Parameters:
        model: PyTorch CNN model (only for age classification).
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for testing data.
        optimizer: Optimizer for the model.
        age_loss_fn: Loss function for age classification.
        epochs: Number of epochs to train the model.
        device: Device for computation (e.g., "cuda" or "cpu").

    Returns:
        results: Dictionary containing training and testing losses and accuracies.
    """
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(epochs):

        # Training Step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            age_loss_fn=age_loss_fn,
            optimizer=optimizer,
            device=device
        )

        # Testing Step
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            age_loss_fn=age_loss_fn,
            device=device
        )

        # Logging
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4%}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4%}")

        # Append Results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results
