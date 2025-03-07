import torch
from timeit import default_timer as timer
from src.model import AgePredictionCNN
from src.train import train
from src.evaluate import plot_loss_and_accuracy_curves, predict, evaluate_predictions
from src.data import AgeDatasetManager
from src.config import device
import torch.nn as nn
from torchvision import transforms




# Define transformation for training and testing data
train_transform = transforms.Compose([
                transforms.Resize(size=(128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

test_transform = transforms.Compose([
                transforms.Resize(size=(128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


# Initialize dataset manager and get dataloaders
dataset_manager = AgeDatasetManager(age_bins=[0, 18, 25, 70], 
                                    age_labels=['Not Old Enough', 'Check ID', 'Old Enough'])
train_loader, val_loader, test_loader = dataset_manager.get_data_loaders(train_size=0.7, 
                                                                         val_size=0.15, 
                                                                         test_size=0.15, 
                                                                         batch_size=16, 
                                                                         train_transform=train_transform, 
                                                                         test_transform=test_transform)
age_idx = {label:i for i, label in enumerate(dataset_manager.age_labels)}

# Initialize model
CNN_model = AgePredictionCNN(num_age_bins=len(age_idx)).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=0.001)
age_loss_fn = torch.nn.MSELoss()

# imbalance in the dataset
class_weights = torch.tensor([5.15, 7.37, 1.49]).to("cpu")
age_loss_fn = nn.CrossEntropyLoss(weight=class_weights)


# Define training parameters
EPOCHS = 10

# Start timing
start_time = timer()

# Train the model
model_results = train(
    model=CNN_model,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    optimizer=optimizer,
    age_loss_fn=age_loss_fn,
    epochs=EPOCHS,
    device=device
)

# End timing
end_time = timer()
print(f'Total Training Time: {end_time - start_time:.3f} seconds')

# Evaluate the model
plot_loss_and_accuracy_curves(model_results)
eval_results = predict(model=CNN_model, dataloader=test_loader)
class_report = evaluate_predictions(eval_results, age_idx)
print(class_report)

print("Done!")

