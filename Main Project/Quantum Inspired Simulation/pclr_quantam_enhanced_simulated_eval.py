import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = {
    "data_dir": "processed_ptbxl",
    "ckpt": "finetuned.pth",
    "save_dir": "quantum_results",

    "in_channels": 12,
    "base": 256,
    "num_classes": 5,
    "signal_length": 5000,
    "batch_size": 64,

    "noise_levels": [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
    "noise_std": 0.03,

    "tta": True,
    "n_aug": 8
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
class_names = ["NORM", "MI", "STTC", "CD", "HYP"]


class DatasetECG(Dataset):
    def __init__(self, root, factor):
        root = Path(root)
        self.x = np.load(root / "test_signals.npy", mmap_mode="r")
        self.y = np.load(root / "test_labels.npy", mmap_mode="r")
        self.factor = factor
        self.std = config["noise_std"] * factor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig = self.x[i].copy()
        T = sig.shape[0]

        if T >= config["signal_length"]:
            s = (T - config["signal_length"]) // 2
            sig = sig[s:s + config["signal_length"]]
        else:
            pad = config["signal_length"] - T
            sig = np.pad(sig, ((0, pad), (0, 0)), mode="edge")

        sig = torch.tensor(sig, dtype=torch.float32).transpose(0, 1)

        if self.factor < 1:
            sig += self.std * torch.randn_like(sig)

        return sig, int(self.y[i])


class Block(nn.Module):
    def __init__(self, c1, c2, s=1):
        super().__init__()
        self.c1 = nn.Conv1d(c1, c2, 7, s, 3)
        self.b1 = nn.BatchNorm1d(c2)
        self.c2 = nn.Conv1d(c2, c2, 7, 1, 3)
        self.b2 = nn.BatchNorm1d(c2)
        self.r = nn.ReLU(True)

        self.down = None
        if s != 1 or c1 != c2:
            self.down = nn.Sequential(
                nn.Conv1d(c1, c2, 1, s),
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
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(enc.dim, config["num_classes"])

    def forward(self, x):
        return self.fc(self.enc(x))


def load_model():
    enc = Encoder(config["in_channels"], config["base"])
    model = Model(enc).to(device)

    ckpt = torch.load(config["ckpt"], map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def aug(x):
    if np.random.rand() < 0.5:
        x += 0.005 * torch.randn_like(x)
    if np.random.rand() < 0.3:
        x *= np.random.uniform(0.95, 1.05)
    if np.random.rand() < 0.3:
        x = torch.roll(x, np.random.randint(-30, 30), dims=1)
    return x


def evaluate(model, loader):
    preds, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)

            if config["tta"]:
                outs = []
                for _ in range(config["n_aug"]):
                    xa = torch.stack([aug(i.clone()) for i in x])
                    outs.append(model(xa))
                out = torch.stack(outs).mean(0)
            else:
                out = model(x)

            p = out.argmax(1).cpu()
            preds.append(p)
            labels.append(y)

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    return preds, labels


def plot_cm(preds, labels, name):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(name)
    plt.savefig(f"{config['save_dir']}/{name}.png")
    plt.close()


def main():
    model = load_model()
    results = []

    for f in config["noise_levels"]:
        print("\nnoise:", f)

        ds = DatasetECG(config["data_dir"], f)
        loader = DataLoader(ds, batch_size=config["batch_size"])

        preds, labels = evaluate(model, loader)

        acc = accuracy_score(labels, preds)
        bacc = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")

        print(f"acc={acc*100:.2f}, bacc={bacc*100:.2f}, f1={f1*100:.2f}")

        plot_cm(preds, labels, f"cm_{f}")

        results.append([f, acc, bacc, f1])

    df = pd.DataFrame(results, columns=["noise", "acc", "bacc", "f1"])
    df.to_csv(f"{config['save_dir']}/summary.csv", index=False)

    print("\nDone")


if __name__ == "__main__":
    main()