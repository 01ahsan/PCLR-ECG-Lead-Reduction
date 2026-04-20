import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

base_dir = "data"
hdf5_path = os.path.join(base_dir, "ecg_tracings.hdf5")
meta_path = os.path.join(base_dir, "attributes.csv")
annot_dir = os.path.join(base_dir, "annotations")
save_dir = "ribeiro_pclr_harmonized"

os.makedirs(save_dir, exist_ok=True)

labels = ["1dAVb", "AF", "LBBB", "RBBB", "SB", "ST"]
target_len = 5000
eps = 1e-7

print("Loading ECG...")
with h5py.File(hdf5_path, "r") as h5:
    signals = h5["tracings"][:].astype(np.float32)

print("Raw ECG:", signals.shape)

pad_width = target_len - signals.shape[1]
if pad_width > 0:
    signals = np.pad(signals, ((0, 0), (0, pad_width), (0, 0)), mode="edge")

print("Padded ECG:", signals.shape)

metadata = pd.read_csv(meta_path)
metadata.index = np.arange(len(metadata))

annot_files = sorted(f for f in os.listdir(annot_dir) if f.endswith(".csv"))
annot_dfs = []

for file_name in annot_files:
    file_path = os.path.join(annot_dir, file_name)
    df = pd.read_csv(file_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.reindex(columns=labels, fill_value=0)
    annot_dfs.append(df)

annot_stack = np.stack([df.values for df in annot_dfs], axis=1)
majority = np.round(annot_stack.mean(axis=1)).astype(int)
y = majority.argmax(axis=1)

mean = np.mean(signals, axis=(0, 1), keepdims=True)
std = np.std(signals, axis=(0, 1), keepdims=True) + eps
signals = (signals - mean) / std

print("Mean per lead:", np.mean(signals, axis=(0, 1)))
print("STD per lead:", np.std(signals, axis=(0, 1)))

indices = np.arange(len(y))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    stratify=y,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=y[temp_idx],
    random_state=42
)

splits = {
    "train": train_idx,
    "val": val_idx,
    "test": test_idx
}

for split_name, idx in splits.items():
    np.save(os.path.join(save_dir, f"{split_name}_signals.npy"), signals[idx])
    np.save(os.path.join(save_dir, f"{split_name}_labels.npy"), y[idx])
    metadata.iloc[idx].to_csv(
        os.path.join(save_dir, f"{split_name}_metadata.csv"),
        index=False
    )
    print(f"Saved {split_name}: {len(idx)} samples")

print("Done")
print("Saved to:", save_dir)