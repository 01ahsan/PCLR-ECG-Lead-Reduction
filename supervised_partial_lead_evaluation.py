import random
import copy
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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
    "save_dir": "results_supervised",

    "in_channels": 12,
    "num_classes": 5,
    "signal_length": 5000,

    "batch_size": 64,
    "epochs": 40,
    "lr": 1e-3,
    "weight_decay": 1e-4,

    "adapter_epochs": 15,
    "adapter_lr": 5e-5
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)

lead_sets = {
    "12": list(range(12)),
    "limb6": list(range(6)),
    "chest6": list(range(6, 12)),
    "3": [0, 1, 6],
    "II": [1],
    "V5": [10]
}


class ECGDataset(Dataset):
    def __init__(self, root, split, leads):
        root = Path(root)
        self.x = np.load(root / f"{split}_signals.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.L = self.x.shape[1]
        self.leads = leads

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig = np.array(self.x[i], dtype=np.float32)

        if self.L >= config["signal_length"]:
            s = (self.L - config["signal_length"]) // 2
            sig = sig[s:s + config["signal_length"]]
        else:
            pad = config["signal_length"] - self.L
            sig = np.pad(sig, ((0, pad), (0, 0)), mode="edge")

        sig = sig[:, self.leads]
        sig = torch.from_numpy(sig).transpose(0, 1)

        return sig, int(self.y[i])


class Block(nn.Module):
    def __init__(self, c1, c2, s=1):
        super().__init__()
        self.c1 = nn.Conv1d(c1, c2, 7, s, 3, bias=False)
        self.b1 = nn.BatchNorm1d(c2)
        self.c2 = nn.Conv1d(c2, c2, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(c2)
        self.r = nn.ReLU(True)

        self.down = None
        if s != 1 or c1 != c2:
            self.down = nn.Sequential(
                nn.Conv1d(c1, c2, 1, s, bias=False),
                nn.BatchNorm1d(c2)
            )

    def forward(self, x):
        idn = x
        out = self.r(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        if self.down is not None:
            idn = self.down(x)
        return self.r(out + idn)


class Model(nn.Module):
    def __init__(self, in_c=12, base=256, n=5):
        super().__init__()
        self.dim = base * 8

        self.stem = nn.Sequential(
            nn.Conv1d(in_c, base, 15, 2, 7, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(True),
            nn.MaxPool1d(3, 2, 1)
        )

        self.l1 = self.layer(base, base, 3)
        self.l2 = self.layer(base, base * 2, 4, 2)
        self.l3 = self.layer(base * 2, base * 4, 6, 2)
        self.l4 = self.layer(base * 4, base * 8, 3, 2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.dim, n)

    def layer(self, c1, c2, n, s=1):
        layers = [Block(c1, c2, s)]
        for _ in range(n - 1):
            layers.append(Block(c2, c2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class Adapter(nn.Module):
    def __init__(self, in_leads):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_leads, 128, 15, 1, 7, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 15, 1, 7, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 12, 15, 1, 7, bias=False),
            nn.BatchNorm1d(12)
        )

    def forward(self, x):
        return self.net(x)


def eval_model(model, loader, adapter=None):
    model.eval()
    if adapter is not None:
        adapter.eval()

    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if adapter is not None:
                x = adapter(x)

            p = model(x).argmax(1).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    return {
        "acc": accuracy_score(labels, preds) * 100,
        "bacc": balanced_accuracy_score(labels, preds) * 100,
        "f1": f1_score(labels, preds, average="macro") * 100
    }


def train_model():
    train = DataLoader(ECGDataset(config["data_dir"], "train", lead_sets["12"]),
                       batch_size=config["batch_size"], shuffle=True)
    val = DataLoader(ECGDataset(config["data_dir"], "val", lead_sets["12"]),
                     batch_size=config["batch_size"])
    test = DataLoader(ECGDataset(config["data_dir"], "test", lead_sets["12"]),
                      batch_size=config["batch_size"])

    model = Model(config["in_channels"], 256, config["num_classes"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    best = 0
    best_state = None

    for e in range(config["epochs"]):
        model.train()
        total = 0

        for x, y in tqdm(train, desc=f"epoch {e+1}"):
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        val_acc = eval_model(model, val)["acc"]
        print(f"epoch {e+1}: loss={total/len(train):.4f}, val={val_acc:.2f}")

        if val_acc > best:
            best = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    test_res = eval_model(model, test)
    print("12-lead:", test_res)

    return model


def train_adapter(base, train_loader, val_loader, n_leads):
    adapter = Adapter(n_leads).to(device)

    for p in base.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(adapter.parameters(), lr=config["adapter_lr"])
    loss_fn = nn.CrossEntropyLoss()

    best = 0
    best_state = None

    for e in range(config["adapter_epochs"]):
        adapter.train()

        for x, y in tqdm(train_loader, desc=f"adapter {e+1}"):
            x, y = x.to(device), y.to(device)

            out = base(adapter(x))
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = eval_model(base, val_loader, adapter)["acc"]
        print(f"adapter epoch {e+1}: {val_acc:.2f}")

        if val_acc > best:
            best = val_acc
            best_state = copy.deepcopy(adapter.state_dict())

    adapter.load_state_dict(best_state)
    return adapter


def main():
    model = train_model()

    results = {}

    for name, leads in lead_sets.items():
        print("\n", name)

        train = DataLoader(ECGDataset(config["data_dir"], "train", leads),
                           batch_size=config["batch_size"], shuffle=True)
        val = DataLoader(ECGDataset(config["data_dir"], "val", leads),
                         batch_size=config["batch_size"])
        test = DataLoader(ECGDataset(config["data_dir"], "test", leads),
                          batch_size=config["batch_size"])

        if name == "12":
            res = eval_model(model, test)
        else:
            adapter = train_adapter(model, train, val, len(leads))
            res = eval_model(model, test, adapter)

        results[name] = res
        print(res)

    print("\nSummary:")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()