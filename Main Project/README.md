# Main Project

## Overview
This directory contains the core implementation of the PCLR-ECG-Lead-Reduction project. It brings together the full workflow for preparing data, training ECG representation models, evaluating performance under lead reduction, and testing cross-dataset generalization.

## Directory Contents
- `Data Preprocessing/` – dataset harmonization and split generation
- `Models/` – supervised and self-supervised ECG model implementations
- `Cross Dataset Evaluation/` – experiments for transfer across datasets
- `Partial Lead Evaluation/` – reduced-lead testing pipelines
- `Quantum Inspired Simulation/` – exploratory simulation extension

## Project Focus
The main project investigates whether meaningful ECG representations can be learned from full-lead data and then remain effective when only a subset of leads is available at inference time. This is especially relevant for wearable and low-cost ECG acquisition systems.

## Experimental Workflow
### Step 1: preprocess data
Raw ECG files are converted into fixed-length, normalized arrays with labels and metadata.

### Step 2: train models
Models are trained using supervised, contrastive, or hybrid training objectives.

### Step 3: evaluate
Performance is measured on standard test sets, reduced-lead subsets, and external datasets.

## Expected Outputs
Depending on the script, outputs may include:
- saved NumPy arrays,
- metadata CSV files,
- trained model checkpoints,
- validation and test accuracy summaries,
- and comparative lead-reduction results.

## Notes
This folder is the best starting point for anyone who wants to understand or reproduce the full experimental pipeline.
