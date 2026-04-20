import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

lead_sets = {
    "12": list(range(12)),
    "limb6": list(range(6)),
    "chest6": list(range(6, 12)),
    "3": [0, 1, 10],
    "II": [1],
    "V5": [10]
}

window = 5000
num_classes = 5


class DatasetECG(Dataset):
    def __init__(self, root, split, leads):
        root = Path(root)
        self.x = np.load(root / f"{split}_signals.npy")
        self.y = np.load(root / f"{split}_labels.npy")
        self.leads = leads

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig = self.x[i]
        y = int(self.y[i])
        T = sig.shape[0]

        if T >= window:
            s = (T - window) // 2
            sig = sig[s:s + window]
        else:
            pad = window - T
            sig = np.pad(sig, ((0, pad), (0, 0)), mode="edge")

        sig = torch.tensor(sig, dtype=torch.float32).transpose(0, 1)

        out = torch.zeros_like(sig)
        out[self.leads] = sig[self.leads]

        return out, y


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

    def forward(self, x):
        h = self.enc(x)
        return self.fc(h)


def load_model(path):
    enc = Encoder()
    model = Model(enc, num_classes).to(device)

    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    model.eval()
    return model


def augment(x):
    if np.random.rand() < 0.7:
        x = x + 0.01 * torch.randn_like(x)
    if np.random.rand() < 0.5:
        x = x * np.random.uniform(0.9, 1.1)
    if np.random.rand() < 0.5:
        x = torch.roll(x, np.random.randint(-50, 50), dims=1)
    if np.random.rand() < 0.3:
        T = x.size(1)
        l = np.random.randint(10, 80)
        s = np.random.randint(0, T - l)
        x[:, s:s+l] = 0
    return x


def evaluate(model, loader, n_aug=10):
    preds, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)

            outs = []
            for _ in range(n_aug):
                xa = torch.stack([augment(i.clone()) for i in x])
                outs.append(model(xa))

            out = torch.stack(outs).mean(0)
            p = out.argmax(1).cpu()

            preds.append(p)
            labels.append(y)

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    return (
        accuracy_score(labels, preds),
        balanced_accuracy_score(labels, preds),
        f1_score(labels, preds, average="macro")
    )


def main():
    data_dir = "processed_ptbxl"
    ckpt = "finetuned.pth"

    model = load_model(ckpt)
    results = {}

    for name, leads in lead_sets.items():
        print("\n", name)

        ds = DatasetECG(data_dir, "test", leads)
        loader = DataLoader(ds, batch_size=64)

        acc, bacc, f1 = evaluate(model, loader, n_aug=12)

        results[name] = (acc, bacc, f1)
        print(f"{name}: {acc*100:.2f}, {bacc*100:.2f}, {f1*100:.2f}")

    print("\nSummary:")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()