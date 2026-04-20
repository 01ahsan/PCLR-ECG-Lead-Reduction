# Models

## Overview
This folder contains the main ECG model implementations used in the project. The code includes supervised baselines, self-supervised contrastive models, and a hybrid model that combines representation learning with classification.

## Included Files
- `Supervised.py`
- `pclr_model.py`
- `simclr.py`
- `Hybrid_Model.py`

## Common Design
Most models in this folder use a 1D convolutional ResNet-style encoder for multi-lead ECG signals. The encoder processes fixed-length ECG windows and produces compact feature embeddings for downstream classification or contrastive learning.

## Models

### 1. Supervised model
`Supervised.py` trains a fully supervised ECG classifier end to end. It uses labeled train, validation, and test splits and reports overall as well as per-class accuracy.

Best use case:
- baseline benchmarking,
- direct comparison with self-supervised approaches,
- and standard full-lead classification experiments.

### 2. PCLR model
`pclr_model.py` implements a patient-aware self-supervised learning pipeline inspired by patient contrastive learning. The training logic groups ECGs by patient identity so that different samples from the same patient can serve as positive pairs in contrastive representation learning.

Best use case:
- learning robust ECG embeddings before fine-tuning,
- exploiting repeated patient recordings,
- and testing representation quality under label scarcity.

### 3. SimCLR model
`simclr.py` provides a SimCLR-style contrastive learning baseline for ECG representation learning using multiple augmented views of the same sample.

Best use case:
- comparing generic contrastive learning against patient-aware contrastive learning,
- measuring the value of augmentation-only positives,
- and establishing a strong self-supervised baseline.

### 4. Hybrid model
`Hybrid_Model.py` combines contrastive projection learning with a supervised classification head. This makes it useful for settings where both representation quality and label-aware optimization matter.

Best use case:
- mixed-objective training,
- representation learning with classification support,
- and improved robustness for downstream reduced-lead experiments.

## Input Expectations
These scripts generally expect preprocessed ECG arrays such as:
- `train_signals.npy`
- `train_labels.npy`
- `val_signals.npy`
- `val_labels.npy`
- `test_signals.npy`
- `test_labels.npy`

Some pipelines also require metadata files for patient-level grouping.

## Outputs
Model scripts may produce:
- trained weights or checkpoints,
- saved embeddings,
- validation summaries,
- final test metrics,
- and result folders for later analysis.

## Notes
If you are starting from scratch, run preprocessing first, then train one baseline and one self-supervised model before moving to partial-lead or transfer evaluation.
