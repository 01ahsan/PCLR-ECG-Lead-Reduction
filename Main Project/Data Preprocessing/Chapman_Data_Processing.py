import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

raw_ecg_dir = "ECGData"
diag_path = "Diagnostics.xlsx"
save_dir = "processed_chapman_harmonized"

os.makedirs(save_dir, exist_ok=True)

target_length = 5000
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

print("Loading Diagnostics.xlsx...")
diag = pd.read_excel(diag_path)
diag["FileName_csv"] = diag["FileName"] + ".csv"

unique_rhythms = sorted(diag["Rhythm"].unique())
rhythm_map = {rhythm: i for i, rhythm in enumerate(unique_rhythms)}
diag["label"] = diag["Rhythm"].map(rhythm_map)

available_files = set(os.listdir(raw_ecg_dir))
diag = diag[diag["FileName_csv"].isin(available_files)]

print("Valid samples:", len(diag))

train_df, test_df = train_test_split(
    diag,
    test_size=0.20,
    stratify=diag["label"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.125,
    stratify=train_df["label"],
    random_state=42
)

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


def load_ecg(filepath):
    df = pd.read_csv(filepath)
    sig = df[leads].values.astype(np.float32)

    length = len(sig)
    if length > target_length:
        sig = sig[:target_length]
    elif length < target_length:
        pad = target_length - length
        sig = np.pad(sig, ((0, pad), (0, 0)), mode="edge")

    return sig


print("Computing lead statistics...")

train_signals = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    file_path = os.path.join(raw_ecg_dir, row["FileName_csv"])
    train_signals.append(load_ecg(file_path))

train_signals = np.stack(train_signals)

lead_means = train_signals.mean(axis=(0, 1))
lead_stds = train_signals.std(axis=(0, 1))
lead_stds[lead_stds < 1e-6] = 1.0

print("Lead means:", lead_means)
print("Lead stds:", lead_stds)


def standardize(sig):
    return (sig - lead_means) / lead_stds


def save_split(df, split_name):
    signals = []
    labels = []
    metadata = []

    print(f"Processing {split_name}...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(raw_ecg_dir, row["FileName_csv"])
        ecg = load_ecg(file_path)
        ecg = standardize(ecg)

        signals.append(ecg)
        labels.append(row["label"])

        metadata.append({
            "file": row["FileName_csv"],
            "rhythm": row["Rhythm"],
            "label": row["label"],
            "patient_id": row["FileName"]
        })

    signals = np.stack(signals)
    labels = np.array(labels, dtype=np.int64)

    np.save(os.path.join(save_dir, f"{split_name}_signals.npy"), signals)
    np.save(os.path.join(save_dir, f"{split_name}_labels.npy"), labels)
    pd.DataFrame(metadata).to_csv(
        os.path.join(save_dir, f"{split_name}_metadata.csv"),
        index=False
    )

    print(f"{split_name} signals:", signals.shape)
    print(f"{split_name} labels:", labels.shape)


save_split(train_df, "train")
save_split(val_df, "val")
save_split(test_df, "test")

print("CHAPMAN PREPROCESSING DONE")