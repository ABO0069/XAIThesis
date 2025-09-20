import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
import warnings
import kagglehub
import pandas as pd
import sys
from tqdm import tqdm
from glob import glob

warnings.filterwarnings('ignore')


# --- 1. Dataset Class and Helper Functions ---
class CustomChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels_dict, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels_dict = labels_dict
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_label_pairs = self._prepare_data()

    def _prepare_data(self):
        pairs = []
        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            label_name = self.labels_dict.get(filename)

            # Skip images with multiple labels for a simpler classification task
            if '|' in str(label_name):
                continue

            if label_name in self.class_to_idx:
                label = self.class_to_idx[label_name]
                pairs.append((img_path, label))
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_path, label = self.image_label_pairs[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# --- 2. Dataset Loading from KaggleHub ---
def load_dataset():
    try:
        print("Downloading the NIH Chest X-ray dataset from Kaggle...")
        path = kagglehub.dataset_download("nih-chest-xrays/data")
        data_dir = str(path)
        print("Dataset downloaded to:", data_dir)

        labels_df = pd.read_csv(os.path.join(data_dir, 'Data_Entry_2017.csv'))
        labels_dict = dict(zip(labels_df['Image Index'], labels_df['Finding Labels']))

        image_paths = glob(os.path.join(data_dir, '**', '*.png'), recursive=True)

        print(f"Found {len(image_paths)} images.")

        unique_labels = sorted(labels_df['Finding Labels'].unique())
        unique_single_labels = [label for label in unique_labels if '|' not in label]
        class_to_idx = {cls: i for i, cls in enumerate(unique_single_labels)}

        class_names = [label.replace('|', '/') for label in unique_single_labels]

        print(f"Number of unique single-labels: {len(unique_single_labels)}")

        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_dataset = CustomChestXrayDataset(
            image_paths=image_paths,
            labels_dict=labels_dict,
            class_to_idx=class_to_idx,
            transform=data_transforms
        )

        return image_dataset, class_names, data_transforms

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have authenticated Kaggle and the dataset is downloaded correctly.")
        return None, None, None


# --- 3. Model Definition and Training ---
def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        print(f'Training Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in tqdm(dataloader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Model training complete.")


# --- 4. Main Execution ---
if __name__ == '__main__':
    model_filename = 'chest_xray_model.pth'

    image_dataset, class_names, data_transforms = load_dataset()
    if image_dataset is None:
        sys.exit(1)

    dataloader = DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=0)
    dataset_size = len(image_dataset)
    print(f"Loaded a real dataset with {dataset_size} images for training.")
    print(f"Training on the following classes: {class_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    print("Starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=2)

    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

    print("\n--- Training and saving complete. You can now use the 'explain_image.py' script. ---")


