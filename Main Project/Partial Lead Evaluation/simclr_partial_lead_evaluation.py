import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = {
    "data_dir": "processed_ptbxl",
    "save_dir": "simclr_partial_eval",

    "model_path": "finetuned_12lead.pth",

    "num_classes": 5,
    "signal_length": 5000,
    "base_filters": 256,
    "batch_size": 64,
    "num_workers": 2
}

lead_sets = {
    "12": list(range(12)),
    "limb6": list(range(6)),
    "chest6": list(range(6, 12)),
    "3": [0, 1, 6],
    "II": [1],
    "V5": [10]
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
Path(f"{config['save_dir']}/cm").mkdir(exist_ok=True)


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


class Encoder(nn.Module):
    def __init__(self, in_c=12, base=256):
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


class Head(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.fc = nn.Linear(d, n)

    def forward(self, x):
        return self.fc(x)


class Model(nn.Module):
    def __init__(self, enc, head):
        super().__init__()
        self.enc = enc
        self.head = head

    def forward(self, x):
        return self.head(self.enc(x))


def pad_to_12(x, leads):
    B, _, T = x.shape
    out = torch.zeros(B, 12, T, device=x.device)

    for i, l in enumerate(leads):
        out[:, l, :] = x[:, i, :]

    return out


def load_model():
    enc = Encoder(12, config["base_filters"]).to(device)
    head = Head(enc.dim, config["num_classes"]).to(device)
    model = Model(enc, head).to(device)

    ckpt = torch.load(config["model_path"], map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


def evaluate(model, loader, leads):
    preds, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            x = pad_to_12(x, leads)
            p = model(x).argmax(1)

            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    acc = accuracy_score(labels, preds) * 100
    bacc = balanced_accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds, average="macro") * 100
    cm = confusion_matrix(labels, preds)

    return acc, bacc, f1, cm


def plot_cm(cm, name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{config['save_dir']}/cm/{name}.png")
    plt.close()


def main():
    model = load_model()
    results = []

    for name, leads in lead_sets.items():
        print("\n", name)

        ds = ECGDataset(config["data_dir"], "test", leads)
        loader = DataLoader(ds, batch_size=config["batch_size"], num_workers=config["num_workers"])

        acc, bacc, f1, cm = evaluate(model, loader, leads)

        print(f"acc={acc:.2f}, bacc={bacc:.2f}, f1={f1:.2f}")

        plot_cm(cm, name)

        results.append({
            "lead": name,
            "n": len(leads),
            "acc": acc,
            "bacc": bacc,
            "f1": f1
        })

    df = pd.DataFrame(results)
    df.to_csv(f"{config['save_dir']}/summary.csv", index=False)

    print("\nSummary:")
    print(df)


if __name__ == "__main__":
    main()