import random, copy, json
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
    "ckpt": "finetune_best.pth",
    "target_root": "processed_target",
    "target_name": "target_dataset",
    "num_classes": 5,

    "save_dir": "cross_eval",

    "batch_size": 64,
    "signal_length": 5000,
    "epochs": 20,
    "lr": 5e-4,
    "wd": 1e-4,
    "patience": 12,

    "in_channels": 12,
    "base": 256,
    "amp": False
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)


class ECGDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.x = np.load(root / f"{split}_signals.npy")
        self.y = np.load(root / f"{split}_labels.npy")
        self.split = split

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig = self.x[i]
        y = int(self.y[i])
        T = sig.shape[0]

        if T > config["signal_length"]:
            if self.split == "train":
                s = np.random.randint(0, T - config["signal_length"] + 1)
            else:
                s = (T - config["signal_length"]) // 2
            sig = sig[s:s + config["signal_length"]]
        else:
            pad = config["signal_length"] - T
            sig = np.pad(sig, ((0, pad), (0, 0)), mode="edge")

        sig = torch.tensor(sig, dtype=torch.float32).transpose(0, 1)

        if self.split == "train":
            if np.random.rand() < 0.7:
                sig += 0.02 * torch.randn_like(sig)
            if np.random.rand() < 0.5:
                sig *= np.random.uniform(0.9, 1.1)

        return sig, y


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
        if self.down:
            idn = self.down(x)
        return self.r(out + idn)


class Encoder(nn.Module):
    def __init__(self, in_c=12, base=256):
        super().__init__()
        self.dim = base * 8

        self.stem = nn.Sequential(
            nn.Conv1d(in_c, base, 15, 2, 7),
            nn.BatchNorm1d(base),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1)
        )

        self.l1 = self.layer(base, base, 3)
        self.l2 = self.layer(base, base * 2, 4, 2)
        self.l3 = self.layer(base * 2, base * 4, 6, 2)
        self.l4 = self.layer(base * 4, base * 8, 3, 2)

        self.pool = nn.AdaptiveAvgPool1d(1)

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
        return self.pool(x).squeeze(-1)


class Model(nn.Module):
    def __init__(self, enc, n):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(enc.dim, n)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        with torch.no_grad():
            h = self.enc(x)
        return self.fc(self.drop(h))


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = model(x).argmax(1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    acc = (preds == labels).mean()

    return acc, preds, labels


def class_weights(y, n):
    y = torch.tensor(y)
    counts = torch.bincount(y, minlength=n).float()
    w = 1.0 / (counts + 1e-6)
    return (w / w.sum() * n)


def main():
    train = ECGDataset(config["target_root"], "train")
    val = ECGDataset(config["target_root"], "val")
    test = ECGDataset(config["target_root"], "test")

    loader = lambda d, s: DataLoader(d, batch_size=s, shuffle=(d.split == "train"))

    train_loader = loader(train, config["batch_size"])
    val_loader = loader(val, config["batch_size"])
    test_loader = loader(test, config["batch_size"])

    enc = Encoder(config["in_channels"], config["base"])
    ckpt = torch.load(config["ckpt"], map_location=device)

    state = {k.replace("encoder.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    enc.load_state_dict(state)

    for p in enc.parameters():
        p.requires_grad = False

    model = Model(enc, config["num_classes"]).to(device)

    weights = class_weights(train.y, config["num_classes"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])

    best, best_state, patience = 0, None, 0

    for e in range(config["epochs"]):
        model.train()
        total = 0

        for x, y in tqdm(train_loader, desc=f"epoch {e+1}"):
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        acc, _, _ = evaluate(model, val_loader)
        print(f"epoch {e+1}: loss={total/len(train_loader):.4f}, val={acc*100:.2f}")

        if acc > best:
            best = acc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config["patience"]:
                break

    model.load_state_dict(best_state)

    acc, preds, labels = evaluate(model, test_loader)
    print(f"\nTest acc: {acc*100:.2f}")

    results = {"acc": float(acc)}
    with open(Path(config["save_dir"]) / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()