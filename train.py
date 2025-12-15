"""
train.py

Purpose:
Train the final CNN model for histopathologic cancer detection
and save the trained model to disk for later deployment.
"""

# =========================
# Imports
# =========================
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================
# Configuration
# =========================
DATA_DIR = "data/images_subset"
LABELS_PATH = "data/subset_labels.csv"
MODEL_PATH = "model_cnn_v1.pth"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


# =========================
# Dataset Definition
# =========================
class CancerDataset(Dataset):
    """
    Custom PyTorch Dataset for loading histopathologic images
    and corresponding cancer labels.
    """

    def __init__(self, labels_df, image_dir):
        self.labels = labels_df
        self.image_dir = image_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['id']}.tif")

        # load image from disk
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize pixel values
        img = img / 255.0

        # convert to tensor and reorder dimensions (C, H, W)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # label tensor
        label = torch.tensor(row["label"], dtype=torch.long)

        return img, label


# =========================
# CNN v1 Model Definition
# =========================
class SimpleCNN(nn.Module):
    """
    Baseline CNN architecture (final selected model).
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # conv → relu → pool
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # flatten feature maps
        x = x.view(x.size(0), -1)

        # fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# =========================
# Training Function
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # reset gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(imgs)

        # compute loss
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss


# =========================
# Main Training Logic
# =========================
def main():
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load labels
    labels_df = pd.read_csv(LABELS_PATH)

    # train/validation split
    train_df, val_df = train_test_split(
        labels_df,
        test_size=0.2,
        random_state=42,
        stratify=labels_df["label"]
    )

    # datasets
    train_dataset = CancerDataset(train_df, DATA_DIR)
    val_dataset = CancerDataset(val_df, DATA_DIR)

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # initialize model
    model = SimpleCNN().to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}"
        )

    # save trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()
