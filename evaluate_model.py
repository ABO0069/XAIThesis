import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.metrics import classification_report
import kagglehub
from PIL import Image
from xai_explanation_2 import CustomChestXrayDataset


def load_and_evaluate():
    model_filename = 'chest_xray_model.pth'

    if not os.path.exists(model_filename):
        print(f"Error: Model file '{model_filename}' not found.")
        print("Please run the training script ('xai_explanation_2.py') first to generate this file.")
        sys.exit(1)

    print("Downloading the NIH Chest X-ray dataset from Kaggle...")
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    data_dir = str(path)
    print("Dataset downloaded to:", data_dir)

    labels_df = pd.read_csv(os.path.join(data_dir, 'Data_Entry_2017.csv'))
    labels_dict = dict(zip(labels_df['Image Index'], labels_df['Finding Labels']))

    image_paths = glob(os.path.join(data_dir, '**', '*.png'), recursive=True)

    unique_labels = sorted(labels_df['Finding Labels'].unique())
    unique_single_labels = [label for label in unique_labels if '|' not in label]
    class_names = [label.replace('|', '/') for label in unique_single_labels]
    class_to_idx = {cls: i for i, cls in enumerate(unique_single_labels)}

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

    # Split the dataset to get a validation set
    train_size = int(0.8 * len(image_dataset))
    val_size = len(image_dataset) - train_size
    _, val_dataset = random_split(image_dataset, [train_size, val_size])

    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Model Performance Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))


if __name__ == '__main__':
    load_and_evaluate()

