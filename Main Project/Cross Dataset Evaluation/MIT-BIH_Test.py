import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

import matplotlib.pyplot as plt


# ================================================================
# CONFIG
# ================================================================
CONFIG = {
    "pclr_ckpt_path": "checkpoints/pclr_model.pth",
    "mitbih_path": "data/mitbih",
    "save_dir": "results",

    "in_channels": 12,
    "base_filters": 256,
    "signal_length": 5000,

    "batch_size": 64,
    "num_workers": 0,
    "n_clusters_range": [3, 5, 7, 10],

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

Path(CONFIG["save_dir"]).mkdir(parents=True, exist_ok=True)


# ================================================================
# DATASET
# ================================================================
class MITBIHDataset(Dataset):
    def __init__(self, data_path, lead_indices=[0, 1], window_length=5000):
        self.data_path = Path(data_path)
        self.lead_indices = lead_indices
        self.window_length = window_length

        self.signals = np.load(self.data_path / "signals.npy", mmap_mode="r")
        self.metadata = pd.read_csv(self.data_path / "metadata.csv")

        self.signal_length = self.signals.shape[1]
        self.indices = np.arange(len(self.signals))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        signal = np.array(self.signals[idx], dtype=np.float32)

        if self.signal_length >= self.window_length:
            start = (self.signal_length - self.window_length) // 2
            signal = signal[start:start + self.window_length]
        else:
            pad_len = self.window_length - self.signal_length
            signal = np.pad(signal, ((0, pad_len), (0, 0)), mode="edge")

        signal = signal[:, self.lead_indices]
        signal = torch.from_numpy(signal).transpose(0, 1)

        return signal, idx


# ================================================================
# MODEL
# ================================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, base_filters=256):
        super().__init__()

        self.embedding_dim = base_filters * 8

        self.conv1 = nn.Conv1d(in_channels, base_filters, 15, 2, 7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(3, 2, 1)

        self.layer1 = self._make_layer(base_filters, base_filters, 3)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, 4, stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, 6, stride=2)
        self.layer4 = self._make_layer(base_filters * 4, base_filters * 8, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        downsample = None

        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm1d(out_c),
            )

        layers = [ResidualBlock1D(in_c, out_c, stride, downsample)]
        for _ in range(blocks - 1):
            layers.append(ResidualBlock1D(out_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class TwoLeadAdapter(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.embedding_dim = encoder.embedding_dim

        old_conv = encoder.conv1

        self.conv1 = nn.Conv1d(
            2,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            w = old_conv.weight.mean(dim=1, keepdim=True)
            self.conv1.weight.copy_(w.repeat(1, 2, 1))

        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.pool = encoder.pool

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.avgpool = encoder.avgpool

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


# ================================================================
# EMBEDDING
# ================================================================
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for signals, _ in tqdm(loader):
            signals = signals.to(device)
            emb = model(signals)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ================================================================
# CLUSTERING
# ================================================================
def evaluate_clustering(embeddings, n_clusters_list):
    results = []

    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        results.append({
            "k": k,
            "silhouette": silhouette_score(embeddings, labels),
            "calinski": calinski_harabasz_score(embeddings, labels),
            "davies": davies_bouldin_score(embeddings, labels),
        })

    return pd.DataFrame(results)


# ================================================================
# VISUALIZATION
# ================================================================
def visualize_embeddings(embeddings, save_dir):
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    emb_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(emb_pca[:, 0], emb_pca[:, 1], s=10, alpha=0.5)
    plt.title("PCA")

    plt.subplot(1, 2, 2)
    plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=10, alpha=0.5)
    plt.title("t-SNE")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/embeddings.png")
    plt.close()


# ================================================================
# MAIN
# ================================================================
def main():
    device = CONFIG["device"]

    dataset = MITBIHDataset(CONFIG["mitbih_path"])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"])

    encoder = ResNet1D(CONFIG["in_channels"], CONFIG["base_filters"])
    ckpt = torch.load(CONFIG["pclr_ckpt_path"], map_location=device)

    encoder.load_state_dict(ckpt["model_state_dict"])

    model = TwoLeadAdapter(encoder).to(device)

    embeddings = extract_embeddings(model, loader, device)

    clustering = evaluate_clustering(embeddings, CONFIG["n_clusters_range"])
    print(clustering)

    visualize_embeddings(embeddings, CONFIG["save_dir"])


if __name__ == "__main__":
    main()