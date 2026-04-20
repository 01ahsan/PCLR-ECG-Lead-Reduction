# Cross Dataset Evaluation

## Overview
This folder contains scripts for evaluating how well learned ECG representations generalize across datasets. Rather than measuring performance only on the source training distribution, these experiments test whether the model transfers to external datasets with different populations, annotation schemes, or acquisition characteristics.

## Included Files
- `MIT-BIH_Test.py`
- `pclr_CrossVal_Cinc2020.py`
- `pclr_CrossVal_Ribeiro.py`
- `pclr_CrossVal_chapman.py`

## Motivation
Strong in-dataset performance does not guarantee real-world robustness. Cross-dataset evaluation is essential for testing whether a model has learned clinically meaningful and transferable signal features instead of dataset-specific shortcuts.

## What This Folder Tests
These scripts are intended to measure:
- out-of-distribution generalization,
- transfer performance across ECG sources,
- and the stability of learned representations under domain shift.

## Script Descriptions

### `MIT-BIH_Test.py`
Runs model evaluation on the MIT-BIH setting or a comparable external benchmark split. This script is useful for assessing how well a trained model transfers beyond its original development dataset.

### `pclr_CrossVal_Cinc2020.py`
Evaluates a PCLR-based model in a cross-validation or transfer setup involving the CinC 2020 benchmark.

### `pclr_CrossVal_Ribeiro.py`
Runs transfer-style evaluation using the Ribeiro-formatted dataset.

### `pclr_CrossVal_chapman.py`
Runs transfer-style evaluation using the Chapman dataset.

## Typical Workflow
1. Train or load a pretrained encoder.
2. Prepare the target dataset in the expected format.
3. Run the corresponding evaluation script.
4. Record accuracy, balanced performance, or other summary metrics.
5. Compare transfer behavior across datasets.

## Why It Matters
Cross-dataset results are especially important in ECG machine learning because signal quality, labeling policies, patient demographics, and device settings often vary substantially between collections.

## Notes
- Ensure label mappings are aligned before comparison.
- Verify signal shape and preprocessing compatibility across datasets.
- Report both average performance and failure cases where possible.
