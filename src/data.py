# Description: This file contains the AgeDataset class and AgeDatasetManager class. The AgeDataset class is used to load images and labels, while the AgeDatasetManager class is used to manage the dataset and create data loaders for training, validation, and testing.
import pandas as pd
# import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from config import CSV_PATH
from pathlib import Path


class AgeDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        PyTorch Dataset class for loading images and labels.

        :param data: Pandas DataFrame containing full image paths and labels.
        :param transform: Transformations to apply to images.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["path"]  # Uses full path from CSV
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["age_label"]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data Preprocessing
class AgeDatasetManager:
    def __init__(self, csv_path=CSV_PATH, age_bins=None, age_labels=None):
        """
        Initializes the dataset manager based on ageutk_data.csv.

        :param csv_path: Path to the CSV file containing full image paths and metadata.
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.age_bins = age_bins
        self.age_labels = age_labels

        # Process data immediately
        self.clean_data()
        self.create_class_label()

    def clean_data(self):
        """
        Ensures the dataset is clean by capping max age at 70 and fixing file paths.
        """
        self.data.loc[self.data["age"] > 70, "age"] = 70

        # Fix incorrect paths
        BASE_DIR = Path(__file__).resolve().parent.parent  # MLGroup105 directory
        correct_image_dir = BASE_DIR / "data" / "organised_images"

        # Update the "path" column to ensure all paths start from the correct directory
        self.data["path"] = self.data["path"].apply(lambda x: str(correct_image_dir / Path(x).name))

        # # Debugging: Print first few paths to verify
        # print("Updated image paths:")
        # print(self.data["path"].head())


    def create_class_label(self):
        """
        Creates age category labels based on predefined bins.

        :param age_bins: List defining the age groups.
        """
        if self.age_bins is None:
            self.age_bins = [0, 18, 25, 70]  # Same bins used in your notebook
        if self.age_labels is None:
            self.age_labels = ['Not Old Enough', 'Check ID', 'Old Enough']
            # age_labels = range(len(self.age_bins) - 1)

        self.data["age_label"] = pd.cut(self.data["age"], bins=self.age_bins, labels=range(len(self.age_bins) - 1))

    def get_data_loaders(self, train_size=0.7, val_size=0.15, test_size=0.15, batch_size=16, 
                         train_transform=None, test_transform=None):
        """
        Splits the dataset into training, validation, and test sets and returns DataLoaders.

        :param train_size: Proportion of the dataset for training.
        :param val_size: Proportion for validation.
        :param test_size: Proportion for testing.
        :param batch_size: Number of samples per batch.
        :param train_transform: Transformations for training set.
        :param test_transform: Transformations for test set.
        :return: train_loader, val_loader, test_loader
        """
        assert train_size + val_size + test_size == 1, "Dataset splits must sum to 1."

        if train_transform is None:        # Define transformations
            train_transform = transforms.Compose([
                transforms.Resize(size=(128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.Resize(size=(128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Create datasets with different transforms
        dataset = AgeDataset(self.data, transform=None)  # No transform yet
        train_len = int(train_size * len(dataset))
        val_len = int(val_size * len(dataset))
        test_len = len(dataset) - train_len - val_len

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

        # Apply different transforms to train and test/val datasets
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = test_transform
        test_dataset.dataset.transform = test_transform

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

