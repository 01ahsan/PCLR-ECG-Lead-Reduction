## 📌 Overview
This repository contains code and experiment resources for studying ECG lead reduction with representation learning and deep learning. The goal is to identify whether diagnostic performance can be preserved when models are trained or evaluated with fewer than the standard 12 ECG leads.

The project is organized around a central experimental pipeline that includes:
- data preprocessing and harmonization,
- multiple model training strategies,
- cross-dataset evaluation,
- partial-lead evaluation,
- and a quantum-inspired simulation extension.

## Repository Structure
- `Main Project/` – primary project workspace
- `Main Project/Data Preprocessing/` – scripts for preparing ECG datasets
- `Main Project/Models/` – supervised, SimCLR, PCLR, and hybrid model implementations
- `Main Project/Cross Dataset Evaluation/` – experiments for testing generalization across datasets
- `Main Project/Partial Lead Evaluation/` – experiments using reduced lead subsets
- `Main Project/Quantum Inspired Simulation/` – exploratory simulation-based evaluation pipeline

## Research Goal
Standard 12-lead ECGs provide rich diagnostic information, but reduced-lead systems are more practical for wearable, portable, and resource-constrained settings. This repository investigates whether learned ECG representations can retain enough signal quality and class-discriminative information to support reliable classification under lead reduction.

## Main Components
### 1. Data preprocessing
The preprocessing stage converts raw ECG sources into harmonized NumPy-based train, validation, and test splits. Signals are padded or cropped to a fixed length, standardized, and paired with labels and metadata.

### 2. Model development
The repository includes:
- a fully supervised 1D ResNet-style ECG classifier,
- a PCLR-style patient-aware self-supervised representation learner,
- a SimCLR-style contrastive baseline,
- and a hybrid model combining contrastive and supervised objectives.

### 3. Evaluation
The codebase evaluates:
- standard in-dataset classification performance,
- robustness under reduced lead configurations,
- and cross-dataset transfer behavior.

## Datasets
The repository appears to support multiple ECG datasets, including harmonized processing for Chapman and Ribeiro-style data, along with evaluation scripts for additional benchmark settings.

Because medical ECG datasets often have access restrictions, raw data may need to be obtained separately from their original sources before running the pipeline.

## Typical Workflow
1. Prepare and harmonize the ECG dataset.
2. Train one or more baseline or self-supervised models.
3. Fine-tune or evaluate the learned encoder.
4. Run reduced-lead evaluation.
5. Run cross-dataset testing to assess generalization.

## 📚 Dependencies
Typical dependencies include:
- Python
- NumPy
- pandas
- PyTorch
- scikit-learn
- tqdm
- Matplotlib
- h5py

## Reproducibility Notes
To reproduce the experiments:
1. place the raw ECG files in the expected folder structure,
2. run the appropriate preprocessing script,
3. train the desired model,
4. evaluate on full-lead and reduced-lead settings,
5. compare results across datasets and experimental variants.

## Disclaimer
This repository is intended for research and experimentation. It should not be used for direct clinical decision-making without proper validation, regulatory review, and domain expert oversight.

## License
This project is distributed under the MIT License unless otherwise noted.



##  Acknowledgements

We acknowledge the developers of the open-source libraries used in this work.

