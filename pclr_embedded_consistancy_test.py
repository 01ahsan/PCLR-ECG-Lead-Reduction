import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mannwhitneyu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = {
    "data_dir": "processed_ptbxl",
    "ckpt": "patient_ssl_best.pth",
    "save_dir": "figures",

    "batch_size": 64,
    "signal_length": 5000,
    "max_pairs": 25000
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)

lead_sets = {
    "12-lead": list(range(12)),
    "Limb-6": list(range(6)),
    "Precordial-6": list(range(6, 12)),
    "3-lead": [0, 1, 6],
    "II": [1],
    "V5": [10]
}


class ECG(Dataset):
    def __init__(self, root, leads):
        root = Path(root)
        self.x = np.load(root / "test_signals.npy", mmap_mode="r")
        meta = pd.read_csv(root / "test_metadata.csv")
        self.pid = meta["patient_id"].astype(str).tolist()
        self.leads = leads

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig = self.x[i]
        T = sig.shape[0]

        if T >= config["signal_length"]:
            s = (T - config["signal_length"]) // 2
            sig = sig[s:s + config["signal_length"]]
        else:
            sig = np.pad(sig, ((0, config["signal_length"]-T),(0,0)), mode="edge")

        sig = torch.tensor(sig, dtype=torch.float32).transpose(0,1)

        out = torch.zeros_like(sig)
        out[self.leads] = sig[self.leads]

        return out, self.pid[i]


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


def load_encoder():
    enc = Encoder().to(device)
    ckpt = torch.load(config["ckpt"], map_location=device)

    state = {
        k.replace("encoder.", ""): v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("encoder.")
    }

    enc.load_state_dict(state)
    enc.eval()

    for p in enc.parameters():
        p.requires_grad = False

    return enc


def get_embeddings(enc, leads):
    ds = ECG(config["data_dir"], leads)
    loader = DataLoader(ds, batch_size=config["batch_size"])

    emb_all, pid_all = [], []

    with torch.no_grad():
        for x, pid in tqdm(loader, leave=False):
            x = x.to(device)
            z = F.normalize(enc(x), dim=1)
            emb_all.append(z.cpu())
            pid_all.extend(pid)

    return torch.cat(emb_all), np.array(pid_all)


def distances(emb, pid):
    intra, inter = [], []
    N = len(emb)

    for _ in range(config["max_pairs"]):
        i, j = random.sample(range(N), 2)
        d = 1 - F.cosine_similarity(
            emb[i].unsqueeze(0), emb[j].unsqueeze(0)
        ).item()

        if pid[i] == pid[j]:
            intra.append(d)
        else:
            inter.append(d)

    return np.array(intra), np.array(inter)


def main():
    enc = load_encoder()
    results = {}

    for name, leads in lead_sets.items():
        print("\n", name)

        emb, pid = get_embeddings(enc, leads)
        intra, inter = distances(emb, pid)

        p = mannwhitneyu(intra, inter, alternative="less").pvalue
        results[name] = (intra, inter)

        print(f"p = {p:.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()

    for ax, (name, (intra, inter)) in zip(axes, results.items()):
        ax.boxplot([intra, inter], labels=["Intra", "Inter"], showfliers=False)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Cosine Distance")

    plt.tight_layout()
    plt.savefig(f"{config['save_dir']}/embedding_consistency.png", dpi=300)
    plt.show()

    print("\nDone")


if __name__ == "__main__":
    main()