import copy
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

config = {
    "data_dir": "processed_ptbxl",
    "save_dir": "Results_supervised",
    "in_channels": 12,
    "num_classes": 5,
    "signal_length": 5000,
    "batch_size": 128,
    "num_epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 2,
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)


class ECGDataset(Dataset):
    def __init__(self, data_dir, split, window_length=5000, leads=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.window_length = window_length
        self.leads = leads if leads is not None else list(range(12))

        self.signals = np.load(self.data_dir / f"{split}_signals.npy", mmap_mode="r")
        self.labels = np.load(self.data_dir / f"{split}_labels.npy", mmap_mode="r")
        self.metadata = pd.read_csv(self.data_dir / f"{split}_metadata.csv")
        self.signal_len = self.signals.shape[1]

        print(f"Loaded {split}: {len(self.signals)} samples")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = np.array(self.signals[idx], dtype=np.float32)

        if self.signal_len >= self.window_length:
            start = (self.signal_len - self.window_length) // 2
            signal = signal[start:start + self.window_length]
        else:
            pad = self.window_length - self.signal_len
            signal = np.pad(signal, ((0, pad), (0, 0)), mode="edge")

        signal = signal[:, self.leads]
        signal = torch.from_numpy(signal).transpose(0, 1)

        label = int(self.labels[idx])
        return signal, label


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, base_filters=256, num_classes=5):
        super().__init__()
        self.embedding_dim = base_filters * 8

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.make_layer(base_filters, base_filters, 3, stride=1)
        self.layer2 = self.make_layer(base_filters, base_filters * 2, 4, stride=2)
        self.layer3 = self.make_layer(base_filters * 2, base_filters * 4, 6, stride=2)
        self.layer4 = self.make_layer(base_filters * 4, base_filters * 8, 3, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(blocks - 1):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    per_class_correct = torch.zeros(config["num_classes"])
    per_class_total = torch.zeros(config["num_classes"])

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            for class_idx in range(config["num_classes"]):
                mask = y == class_idx
                per_class_total[class_idx] += mask.sum().item()
                per_class_correct[class_idx] += (preds[mask] == y[mask]).sum().item()

    overall_acc = correct / total
    per_class_acc = (per_class_correct / torch.clamp(per_class_total, min=1)).cpu().numpy()

    return overall_acc, per_class_acc


def train_supervised():
    train_set = ECGDataset(config["data_dir"], "train", window_length=config["signal_length"])
    val_set = ECGDataset(config["data_dir"], "val", window_length=config["signal_length"])
    test_set = ECGDataset(config["data_dir"], "test", window_length=config["signal_length"])

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model = ResNet1D(
        in_channels=config["in_channels"],
        base_filters=256,
        num_classes=config["num_classes"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    print("Starting training...")

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_acc, _ = evaluate(model, val_loader)

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Acc={val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print("Saved new best model")

    print("Testing best model...")

    model.load_state_dict(best_state)
    test_acc, per_class_acc = evaluate(model, test_loader)

    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    for i, acc in enumerate(per_class_acc):
        print(f"Class {i}: {acc * 100:.2f}%")

    return model, best_val_acc, test_acc


if __name__ == "__main__":
    model, best_val_acc, final_test_acc = train_supervised()
    print("Best Val Acc:", best_val_acc * 100)
    print("Final Test Acc:", final_test_acc * 100)