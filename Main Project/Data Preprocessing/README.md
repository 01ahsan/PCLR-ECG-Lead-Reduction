# Data Preprocessing

## Overview
This folder contains preprocessing scripts that convert raw ECG datasets into a unified format suitable for training and evaluation. The scripts standardize signal length, normalize lead values, assign labels, and save train, validation, and test splits.

## Included Scripts
- `Chapman_Data_Processing.py`
- `Ribeiro_Data_Processing.py`

## Purpose
The datasets used in ECG research often differ in file format, annotation structure, and metadata organization. These scripts create a harmonized representation so downstream models can be trained with a consistent data interface.

## Processing Steps
Typical preprocessing operations include:
- reading raw ECG signals from CSV or HDF5 files,
- selecting the standard 12 ECG leads,
- truncating or padding signals to a fixed length,
- computing training-set normalization statistics,
- applying standardization,
- generating train, validation, and test splits,
- and saving labels and metadata.

## Chapman preprocessing
The Chapman pipeline:
- reads ECG CSV files and diagnostic metadata from `Diagnostics.xlsx`,
- maps rhythm labels to numeric classes,
- keeps only samples with available ECG files,
- performs stratified train/validation/test splitting,
- computes per-lead mean and standard deviation from the training split,
- standardizes each signal,
- and saves processed arrays plus metadata.

## Ribeiro preprocessing
The Ribeiro pipeline:
- reads ECG tracings from an HDF5 file,
- loads metadata and annotation CSV files,
- builds class labels from annotation consensus,
- pads signals to a fixed target length,
- standardizes signals,
- performs stratified train/validation/test splitting,
- and saves processed arrays and metadata.

## Output Format
Each preprocessing script saves:
- `train_signals.npy`
- `train_labels.npy`
- `train_metadata.csv`
- `val_signals.npy`
- `val_labels.npy`
- `val_metadata.csv`
- `test_signals.npy`
- `test_labels.npy`
- `test_metadata.csv`

## Usage
Run the appropriate script after placing the raw dataset files in the expected directory structure. Update file paths in the script if your local layout differs.

## Important Notes
- Raw datasets are not bundled with this repository.
- Label definitions depend on the source dataset.
- Always verify class balance and metadata integrity after preprocessing.
