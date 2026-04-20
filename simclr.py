import copy
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    "save_dir": "results",
    "in_channels": 12,
    "num_classes": 5,
    "signal_length": 5000,
    "base_filters": 256,
    "batch_size": 64,
    "epochs_pretrain": 100,
    "epochs_finetune": 20,
    "lr_pretrain": 1e-3,
    "lr_finetune": 1e-4,
    "weight_decay": 1e-4,
    "temperature": 0.1,
    "num_workers": 2
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)


class ECGDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = Path(data_dir)
        self.signals = np.load(self.data_dir / f"{split}_signals.npy", mmap_mode="r")
        self.labels = np.load(self.data_dir / f"{split}_labels.npy", mmap_mode="r")
        self.signal_len = self.signals.shape[1]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = np.array(self.signals[idx], dtype=np.float32)

        if self.signal_len >= config["signal_length"]:
            start = (self.signal_len - config["signal_length"]) // 2
            x = x[start:start + config["signal_length"]]
        else:
            pad = config["signal_length"] - self.signal_len
            x = np.pad(x, ((0, pad), (0, 0)), mode="edge")

        x = torch.from_numpy(x).transpose(0, 1)
        y = int(self.labels[idx])
        return x, y


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(True)

        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, base_filters=256):
        super().__init__()
        self.embedding_dim = base_filters * 8

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, 15, 2, 7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(True),
            nn.MaxPool1d(3, 2, 1)
        )

        self.layer1 = self._layer(base_filters, base_filters, 3)
        self.layer2 = self._layer(base_filters, base_filters * 2, 4, 2)
        self.layer3 = self._layer(base_filters * 2, base_filters * 4, 6, 2)
        self.layer4 = self._layer(base_filters * 4, base_filters * 8, 3, 2)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def _layer(self, in_c, out_c, blocks, stride=1):
        layers = [ResBlock(in_c, out_c, stride)]
        for _ in range(blocks - 1):
            layers.append(ResBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return x


class SimCLR(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        dim = encoder.embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, 512)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


def info_nce(z1, z2, temp):
    logits = torch.matmul(z1, z2.T) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def augment(x):
    if np.random.rand() < 0.5:
        x = x + 0.02 * torch.randn_like(x)
    if np.random.rand() < 0.5:
        x = x * (0.9 + 0.2 * np.random.rand())
    if np.random.rand() < 0.5:
        x = torch.roll(x, shifts=np.random.randint(-50, 50), dims=1)
    return x


def pretrain(model, loader):
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr_pretrain"], weight_decay=config["weight_decay"])

    for epoch in range(config["epochs_pretrain"]):
        model.train()
        total = 0

        for x, _ in tqdm(loader, desc=f"Pretrain {epoch+1}"):
            x = x.to(device)

            x1 = torch.stack([augment(s) for s in x])
            x2 = torch.stack([augment(s) for s in x])

            z1 = model(x1)
            z2 = model(x2)

            loss = info_nce(z1, z2, config["temperature"])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1} loss: {total/len(loader):.4f}")


class Head(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.fc = nn.Linear(dim, n)

    def forward(self, x):
        return self.fc(x)


def finetune(encoder, train_loader, val_loader):
    model = nn.Module()
    model.encoder = encoder
    model.head = Head(encoder.embedding_dim, config["num_classes"])
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr_finetune"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    best = 0
    best_state = None

    for epoch in range(config["epochs_finetune"]):
        model.train()
        total = 0

        for x, y in tqdm(train_loader, desc=f"Finetune {epoch+1}"):
            x, y = x.to(device), y.to(device)

            logits = model.head(model.encoder(x))
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: loss={total/len(train_loader):.4f}, val={acc*100:.2f}%")

        if acc > best:
            best = acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model.head(model.encoder(x)).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


if __name__ == "__main__":
    train_set = ECGDataset(config["data_dir"], "train")
    val_set = ECGDataset(config["data_dir"], "val")
    test_set = ECGDataset(config["data_dir"], "test")

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    encoder = ResNet1D(config["in_channels"], config["base_filters"]).to(device)
    model = SimCLR(encoder).to(device)

    pretrain(model, train_loader)

    encoder = model.encoder
    model = finetune(encoder, train_loader, val_loader)

    test_acc = evaluate(model, test_loader)
    print("Final test accuracy:", test_acc * 100)