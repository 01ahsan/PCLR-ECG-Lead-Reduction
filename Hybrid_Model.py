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
    "save_dir": "pclr_results",

    "batch_size": 32,
    "signal_length": 5000,

    "epochs": 26,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "temperature": 0.2,
    "views": 4,

    "in_channels": 12,
    "base_filters": 256,
    "proj_dim": 512,

    "num_classes": 5,

    "epochs_ft": 24,
    "lr_ft": 1e-4
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)


class ECGDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.x = np.load(root / f"{split}_signals.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.L = self.x.shape[1]

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

        sig = torch.from_numpy(sig).transpose(0, 1)
        return sig, int(self.y[i])


class Augment:
    def __call__(self, x):
        if np.random.rand() < 0.7:
            x = x + 0.03 * torch.randn_like(x)
        if np.random.rand() < 0.7:
            x = x * (0.8 + 0.4 * np.random.rand())
        if np.random.rand() < 0.5:
            x = torch.roll(x, np.random.randint(-100, 100), dims=1)
        return x


def make_views(x, aug, n):
    out = []
    for _ in range(n):
        v = x.clone()
        for i in range(v.size(0)):
            v[i] = aug(v[i])
        out.append(v)
    return out


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


class PCLR(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        d = enc.dim

        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Linear(d, config["proj_dim"])
        )

        self.cls = nn.Linear(d, config["num_classes"])

    def forward(self, x, proj=False):
        h = self.enc(x)
        if proj:
            z = self.proj(h)
            return F.normalize(z, dim=1)
        return h


def supcon_loss(z, y, temp):
    B, V, D = z.shape
    z = F.normalize(z, dim=2)
    z = z.view(B * V, D)

    y = y.repeat_interleave(V)
    sim = torch.matmul(z, z.T) / temp

    mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    mask.fill_diagonal_(0)

    exp = torch.exp(sim)
    pos = (exp * mask).sum(1)
    neg = (exp * (1 - mask)).sum(1)

    valid = pos > 0
    return -torch.log(pos[valid] / (pos[valid] + neg[valid] + 1e-8)).mean()


def train(model, loader):
    aug = Augment()
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    ce = nn.CrossEntropyLoss()

    for e in range(config["epochs"]):
        model.train()
        total = 0

        for x, y in tqdm(loader, desc=f"pclr {e+1}"):
            x, y = x.to(device), y.to(device)

            views = make_views(x, aug, config["views"])
            feats = torch.stack([model(v, proj=True) for v in views], dim=1)

            loss1 = supcon_loss(feats, y, config["temperature"])

            emb = model.enc(x)
            logits = model.cls(emb)
            loss2 = ce(logits, y)

            loss = loss1 + 0.25 * loss2

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch {e+1}: loss={total/len(loader):.4f}")


class Head(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.fc = nn.Linear(d, n)

    def forward(self, x):
        return self.fc(x)


def finetune(enc, train_loader, val_loader):
    model = nn.Module()
    model.enc = enc
    model.head = Head(enc.dim, config["num_classes"])
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr_ft"])
    loss_fn = nn.CrossEntropyLoss()

    best = 0
    best_state = None

    for e in range(config["epochs_ft"]):
        model.train()

        for x, y in tqdm(train_loader, desc=f"ft {e+1}"):
            x, y = x.to(device), y.to(device)

            out = model.head(model.enc(x))
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = evaluate(model, val_loader)
        print(f"epoch {e+1}: val={acc*100:.2f}")

        if acc > best:
            best = acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def evaluate(model, loader):
    model.eval()
    c, t = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model.head(model.enc(x)).argmax(1)
            c += (p == y).sum().item()
            t += y.size(0)

    return c / t


if __name__ == "__main__":
    train_set = ECGDataset(config["data_dir"], "train")
    val_set = ECGDataset(config["data_dir"], "val")
    test_set = ECGDataset(config["data_dir"], "test")

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    enc = Encoder(config["in_channels"], config["base_filters"]).to(device)
    model = PCLR(enc).to(device)

    train(model, train_loader)

    model = finetune(model.enc, train_loader, val_loader)

    test_acc = evaluate(model, test_loader)
    print("Final test acc:", test_acc * 100)