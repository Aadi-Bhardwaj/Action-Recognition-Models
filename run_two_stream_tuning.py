import os
import csv
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.two_stream_dataset import TwoStreamDataset
from models.two_stream import TwoStreamNetwork

# === Paths ===
train_csv = "D:\\ActionRecognition\\splits\\train.csv"
val_csv = "D:\\ActionRecognition\\splits\\val.csv"
rgb_root = "D:\\TwoStream\\data\\rgb_frames"
flow_root = "D:\\TwoStream\\data\\flow_frames"
label_map_path = "D:\\TwoStream\\data\\label_map.json"

# === Load Label Map ===
with open(label_map_path, "r") as f:
    label_map = json.load(f)
num_classes = len(label_map)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

flow_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Datasets ===
train_dataset = TwoStreamDataset(
    train_csv, rgb_root, flow_root, label_map, 
    rgb_transform=rgb_transform, flow_transform=flow_transform
)

val_dataset = TwoStreamDataset(
    val_csv, rgb_root, flow_root, label_map, 
    rgb_transform=rgb_transform, flow_transform=flow_transform
)

# === Hyperparameter Space ===
search_space = {
    "learning_rate": [1e-4, 5e-4],  
    "batch_size": [8, 16],          
    "optimizer": ['SGD','Adam'],           
    "weight_decay": [0, 1e-4],      
}

def random_sample(space):
    return {k: random.choice(v) for k, v in space.items()}

# === Training and Evaluation Function ===
def train_and_evaluate(hparams, trial_id, num_epochs=10, patience=2):
    batch_size = hparams["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = TwoStreamNetwork(num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        print(f"🔧 Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if hparams["optimizer"] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"], momentum=0.9)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # === Train ===
        model.train()
        correct_train, total_train = 0, 0
        train_loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch}", leave=False)
        for rgb, flow, labels in train_loop:
            rgb, flow, labels = rgb.to(device), flow.to(device), labels.to(device)
            outputs = model(rgb, flow)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)
            train_loop.set_postfix(loss=loss.item())

        train_acc = correct_train / total_train

        # === Validate ===
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for rgb, flow, labels in val_loader:
                rgb, flow, labels = rgb.to(device), flow.to(device), labels.to(device)
                outputs = model(rgb, flow)
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        print(f"[Epoch {epoch}] Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        # === Early Stopping Check ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_model_trial_{trial_id}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    return best_val_acc

# === Run Random Search ===
num_trials = 20
results = []

if __name__ == "__main__":  
    csv_path = "hyperparameter_results.csv"
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["trial", "learning_rate", "batch_size", "optimizer", "weight_decay", "val_accuracy"])

        for i in range(num_trials):
            print(f"\nTrial {i+1}/{num_trials}")
            hparams = random_sample(search_space)
            print("Hyperparams:", hparams)

            acc = train_and_evaluate(hparams, trial_id=i+1)
            print(f"Val Accuracy: {acc:.4f}")

            writer.writerow([
                i+1,
                hparams["learning_rate"],
                hparams["batch_size"],
                hparams["optimizer"],
                hparams["weight_decay"],
                acc
            ])

            results.append((hparams, acc))

    # === Show Top Configurations ===
    results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 3 Hyperparameter Configurations:")
    for i, (h, acc) in enumerate(results[:3]):
        print(f"\nRank {i+1} - Acc: {acc:.4f}")
        for k, v in h.items():
            print(f"  {k}: {v}")
