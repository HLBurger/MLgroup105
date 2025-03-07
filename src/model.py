# import torch
import torch.nn as nn
import torch.nn.functional as F


class AgePredictionCNN(nn.Module):
    def __init__(self, num_age_bins=3, input_size=128, sqeeze=False):
        super().__init__()

        self.squeeze = sqeeze

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Corrected fully connected input size
        fc_input_dim = input_size * 16 * 16  # Correct size after conv+pooling

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_age_bins)

    def forward(self, x):
        # Convolution layers with activation and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten for fully connected layers
        x = x.view(x.shape[0], -1)  # Correct flattening

        # Fully connected layers with dropout and activation
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        if self.squeeze:
            x = self.fc3(x).squeeze(-1)
        else:
            # Output layer (no softmax needed with CrossEntropyLoss)
            x = self.fc3(x)
        return x  # Output logits
