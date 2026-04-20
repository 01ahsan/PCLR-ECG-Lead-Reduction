import random, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from tqdm import tqdm
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = {
    "data_dir": "processed_ptbxl",
    "save_dir": "patient_ssl",

    "batch_size": 64,
    "num_workers": 2,
    "signal_length": 5000,

    "epochs_ssl": 100,
    "lr_ssl": 1e-3,
    "wd_ssl": 1e-4,
    "temp": 0.2,
    "views": 4,
    "warmup": 5,

    "in_channels": 12,
    "base": 256,
    "proj_dim": 512,

    "linear_epochs": 400,
    "linear_lr": 1e-2,
    "linear_wd": 1e-4,
    "hidden": 1024,
    "num_classes": 5,

    "epochs_ft": 24,
    "lr_ft": 1e-4,
    "wd_ft": 1e-4
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
Path(f"{config['save_dir']}/checkpoints").mkdir(exist_ok=True)
Path(f"{config['save_dir']}/figures").mkdir(exist_ok=True)


class Aug:
    def __call__(self, x):
        C, T = x.shape

        if np.random.rand() < 0.8:
            x += np.random.uniform(0.01, 0.05) * torch.randn_like(x)

        if np.random.rand() < 0.7:
            x *= np.random.uniform(0.8, 1.2)

        if np.random.rand() < 0.7:
            x = torch.roll(x, np.random.randint(-150, 150), dims=1)

        if np.random.rand() < 0.5:
            x += (np.random.rand() - 0.5) * 0.3

        if np.random.rand() < 0.4:
            x[:, T//2:] *= np.random.uniform(0.9, 1.1)

        if np.random.rand() < 0.3 and C > 3:
            ch = np.random.choice(C, np.random.randint(1, min(4, C)), replace=False)
            x[ch] = 0

        if np.random.rand() < 0.5:
            l = np.random.randint(20, 80)
            s = np.random.randint(0, max(1, T - l))
            x[:, s:s+l] = 0

        return x


def make_views(x, aug, n):
    out = []
    for _ in range(n):
        v = x.clone()
        for i in range(v.size(0)):
            v[i] = aug(v[i])
        out.append(v)
    return out


class ECG(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.x = np.load(root / f"{split}_signals.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        meta = pd.read_csv(root / f"{split}_metadata.csv")

        self.pid = meta["patient_id"].astype(str).tolist()
        self.map = defaultdict(list)

        for i, p in enumerate(self.pid):
            self.map[p].append(i)

        self.patients = list(self.map.keys())
        self.split = split

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig = self.x[i]
        y = int(self.y[i])
        pid = self.pid[i]

        T = sig.shape[0]
        if T >= config["signal_length"]:
            if self.split == "train":
                s = np.random.randint(0, T - config["signal_length"] + 1)
            else:
                s = (T - config["signal_length"]) // 2
            sig = sig[s:s+config["signal_length"]]
        else:
            sig = np.pad(sig, ((0, config["signal_length"]-T),(0,0)), mode="edge")

        sig = torch.tensor(sig, dtype=torch.float32).transpose(0,1)

        return {"signal": sig, "label": y, "pid": pid}


class PatientSampler(Sampler):
    def __init__(self, ds, batch, k=2):
        self.ds = ds
        self.batch = batch
        self.k = k
        self.valid = [p for p,v in ds.map.items() if len(v) >= k]
        self.pb = batch // k

    def __iter__(self):
        pts = self.valid.copy()
        random.shuffle(pts)

        for i in range(0, len(pts), self.pb):
            p = pts[i:i+self.pb]
            if len(p) < self.pb:
                continue

            idx = []
            for pid in p:
                idx += random.sample(self.ds.map[pid], self.k)
            yield idx

    def __len__(self):
        return len(self.valid) // self.pb


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
            self.down = nn.Sequential(nn.Conv1d(c1,c2,1,s), nn.BatchNorm1d(c2))

    def forward(self, x):
        idn = x
        out = self.r(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        if self.down: idn = self.down(x)
        return self.r(out + idn)


class Encoder(nn.Module):
    def __init__(self, c=12, base=256):
        super().__init__()
        self.dim = base*8

        self.stem = nn.Sequential(
            nn.Conv1d(c, base, 15, 2, 7),
            nn.BatchNorm1d(base),
            nn.ReLU(),
            nn.MaxPool1d(3,2,1)
        )

        self.l1 = self.layer(base, base, 3)
        self.l2 = self.layer(base, base*2, 4, 2)
        self.l3 = self.layer(base*2, base*4, 6, 2)
        self.l4 = self.layer(base*4, base*8, 3, 2)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def layer(self,c1,c2,n,s=1):
        l=[Block(c1,c2,s)]
        for _ in range(n-1): l.append(Block(c2,c2))
        return nn.Sequential(*l)

    def forward(self,x):
        x=self.stem(x)
        x=self.l1(x); x=self.l2(x)
        x=self.l3(x); x=self.l4(x)
        return self.pool(x).squeeze(-1)


class Model(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        d = enc.dim
        self.proj = nn.Sequential(
            nn.Linear(d,d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Linear(d, config["proj_dim"])
        )

    def forward(self,x,proj=False):
        h = self.enc(x)
        if proj:
            return F.normalize(self.proj(h), dim=1)
        return h


def loss_fn(z, pid):
    B,V,D = z.shape
    z = F.normalize(z, dim=2).view(B*V, D)

    ids = []
    for p in pid: ids += [p]*V

    sim = torch.matmul(z,z.T)/config["temp"]

    mask = torch.zeros_like(sim)
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i!=j and ids[i]==ids[j]:
                mask[i,j]=1

    exp = torch.exp(sim)
    pos = (exp*mask).sum(1)
    neg = (exp*(1-mask)).sum(1)

    valid = pos>0
    return -torch.log(pos[valid]/(pos[valid]+neg[valid]+1e-8)).mean()


def train_ssl(model, loader):
    aug = Aug()
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr_ssl"], weight_decay=config["wd_ssl"])

    for e in range(config["epochs_ssl"]):
        model.train()
        total = 0

        for b in tqdm(loader, desc=f"ssl {e+1}"):
            x = b["signal"].to(device)
            pid = b["pid"]

            views = make_views(x, aug, config["views"])
            feats = torch.stack([model(v, proj=True) for v in views], dim=1)

            loss = loss_fn(feats, pid)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch {e+1}: {total/len(loader):.4f}")


def main():
    train = ECG(config["data_dir"], "train")

    sampler = PatientSampler(train, config["batch_size"])
    loader = DataLoader(train, batch_sampler=sampler)

    enc = Encoder()
    model = Model(enc).to(device)

    train_ssl(model, loader)


if __name__ == "__main__":
    main()