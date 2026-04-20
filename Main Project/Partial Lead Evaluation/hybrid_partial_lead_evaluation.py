import random, copy
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt

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

    "batch_size": 64,
    "num_workers": 2,
    "signal_length": 5000,

    "epochs_pclr": 26,
    "lr_pclr": 1e-3,
    "wd_pclr": 1e-4,
    "temp": 0.2,
    "views": 4,
    "aux_w": 0.25,

    "in_channels": 12,
    "base": 256,
    "proj_dim": 512,

    "linear_epochs": 400,
    "linear_lr": 1e-2,
    "num_classes": 5,

    "epochs_ft": 24,
    "lr_ft": 1e-4,

    "adapter_epochs": 15,
    "adapter_lr": 5e-5,

    "skip_train": True,
    "finetuned_path": "finetune_best.pth"
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


class DatasetECG(Dataset):
    def __init__(self, root, split="train", leads=None):
        root = Path(root)
        self.x = np.load(root / f"{split}_signals.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.L = self.x.shape[1]
        self.leads = leads if leads is not None else list(range(12))
        meta = pd.read_csv(root / f"{split}_metadata.csv")
        self.pid = meta["patient_id"].astype(str).tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig = self.x[i]
        T = sig.shape[0]

        if T >= config["signal_length"]:
            start = np.random.randint(0, T - config["signal_length"] + 1)
            sig = sig[start:start + config["signal_length"]]
        else:
            pad = config["signal_length"] - T
            sig = np.pad(sig, ((0, pad), (0, 0)), mode="edge")

        sig = sig[:, self.leads]
        sig = torch.tensor(sig, dtype=torch.float32).transpose(0, 1)

        return {"signal": sig, "label": torch.tensor(int(self.y[i]))}


class Aug:
    def __call__(self, x):
        if np.random.rand() < 0.7:
            x += 0.03 * torch.randn_like(x)
        if np.random.rand() < 0.7:
            x *= (0.8 + 0.4 * np.random.rand())
        if np.random.rand() < 0.7:
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
            return F.normalize(self.proj(h), dim=1)
        return h


def supcon(z, y, t):
    B, V, D = z.shape
    z = F.normalize(z, dim=2).view(B * V, D)
    y = y.repeat_interleave(V)

    sim = torch.matmul(z, z.T) / t
    mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    mask.fill_diagonal_(0)

    exp = torch.exp(sim)
    pos = (exp * mask).sum(1)
    neg = (exp * (1 - mask)).sum(1)

    valid = pos > 0
    return -torch.log(pos[valid] / (pos[valid] + neg[valid] + 1e-8)).mean()


def train_pclr(model, loader):
    aug = Aug()
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr_pclr"], weight_decay=config["wd_pclr"])
    ce = nn.CrossEntropyLoss()

    for e in range(config["epochs_pclr"]):
        model.train()
        total = 0

        for b in tqdm(loader, desc=f"pclr {e+1}"):
            x = b["signal"].to(device)
            y = b["label"].to(device)

            views = make_views(x, aug, config["views"])
            feats = torch.stack([model(v, proj=True) for v in views], dim=1)

            loss1 = supcon(feats, y, config["temp"])

            emb = model.enc(x)
            loss2 = ce(model.cls(emb), y)

            loss = loss1 + config["aux_w"] * loss2

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch {e+1}: {total/len(loader):.4f}")


class Adapter(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n, 128, 15, 1, 7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 12, 15, 1, 7),
            nn.BatchNorm1d(12)
        )

    def forward(self, x):
        return self.net(x)


def eval_model(model, loader, adapter=None):
    model.eval()
    if adapter: adapter.eval()

    preds, labels = [], []

    with torch.no_grad():
        for b in loader:
            x = b["signal"].to(device)
            y = b["label"].to(device)

            if adapter:
                x = adapter(x)

            p = model.cls(model.enc(x)).argmax(1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    return {
        "acc": accuracy_score(labels, preds) * 100,
        "bacc": balanced_accuracy_score(labels, preds) * 100,
        "f1": f1_score(labels, preds, average="macro") * 100
    }


def main():
    train = DataLoader(DatasetECG(config["data_dir"], "train"),
                       batch_size=config["batch_size"], shuffle=True)

    val = DataLoader(DatasetECG(config["data_dir"], "val"),
                     batch_size=config["batch_size"])

    test = DataLoader(DatasetECG(config["data_dir"], "test"),
                      batch_size=config["batch_size"])

    enc = Encoder(config["in_channels"], config["base"]).to(device)
    model = PCLR(enc).to(device)

    if not config["skip_train"]:
        train_pclr(model, train)
    else:
        ckpt = torch.load(config["finetuned_path"], map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    results = {}

    for name, leads in lead_sets.items():
        print("\n", name)

        ds = DataLoader(DatasetECG(config["data_dir"], "test", leads),
                        batch_size=config["batch_size"])

        if name == "12":
            res = eval_model(model, ds)
        else:
            adapter = Adapter(len(leads)).to(device)
            res = eval_model(model, ds, adapter)

        results[name] = res
        print(res)

    print("\nSummary:")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()