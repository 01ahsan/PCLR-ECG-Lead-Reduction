import random, copy, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SEED = 42

def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        return torch.device("cuda")
    return torch.device("cpu")

device = setup()
print("Using device:", device)

config = {
    "ckpt": "finetuned.pth",
    "data_dir": "processed_ribeiro",
    "save_dir": "cross_ribeiro",

    "batch_size": 32,
    "signal_length": 5000,

    "in_channels": 12,
    "base": 256,

    "epochs": 20,
    "lr": 5e-4,
    "wd": 1e-4,
    "patience": 15
}

Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)


class ECG(Dataset):
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
            s = np.random.randint(0, T - config["signal_length"] + 1) if self.split=="train" else (T - config["signal_length"])//2
            sig = sig[s:s+config["signal_length"]]
        else:
            sig = np.pad(sig, ((0, config["signal_length"]-T),(0,0)), mode="edge")

        sig = torch.tensor(sig, dtype=torch.float32).transpose(0,1)

        if self.split == "train":
            if random.random() < 0.7:
                sig += 0.02 * torch.randn_like(sig)
            if random.random() < 0.5:
                sig *= random.uniform(0.9, 1.1)

        return sig, torch.tensor(y)


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
    def __init__(self, enc, n):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(enc.dim, n)

    def forward(self, x):
        with torch.no_grad():
            h = self.enc(x)
        return self.fc(h)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu()
            preds.append(p)
            labels.append(y)

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    acc = (preds == labels).mean()
    return acc, preds, labels


def main():
    train = ECG(config["data_dir"], "train")
    val   = ECG(config["data_dir"], "val")
    test  = ECG(config["data_dir"], "test")

    labels_all = np.concatenate([train.y, val.y, test.y])
    unique = np.sort(np.unique(labels_all))
    mapping = {v:i for i,v in enumerate(unique)}

    train.y = np.array([mapping[i] for i in train.y])
    val.y   = np.array([mapping[i] for i in val.y])
    test.y  = np.array([mapping[i] for i in test.y])

    n_classes = len(unique)
    print("Classes:", n_classes)

    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val, batch_size=config["batch_size"])
    test_loader  = DataLoader(test, batch_size=config["batch_size"])

    enc = Encoder(config["in_channels"], config["base"])
    ckpt = torch.load(config["ckpt"], map_location=device)

    state = {k.replace("encoder.",""):v for k,v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    enc.load_state_dict(state)
    enc.to(device).eval()

    model = Model(enc, n_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    loss_fn = nn.CrossEntropyLoss()

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

        val_acc, _, _ = evaluate(model, val_loader)
        print(f"epoch {e+1}: loss={total/len(train_loader):.4f}, val={val_acc*100:.2f}")

        if val_acc > best:
            best = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config["patience"]:
                break

    model.load_state_dict(best_state)

    acc, preds, labels = evaluate(model, test_loader)
    print(f"\nTest acc: {acc*100:.2f}")

    with open(Path(config["save_dir"]) / "results.json", "w") as f:
        json.dump({"acc": float(acc)}, f, indent=2)


if __name__ == "__main__":
    main()