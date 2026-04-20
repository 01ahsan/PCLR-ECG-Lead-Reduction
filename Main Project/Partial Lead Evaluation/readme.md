# Partial Lead Evaluation

## Overview
This folder evaluates model performance when only a subset of ECG leads is available at inference time. These experiments directly support the main goal of the repository: determining how much diagnostic capability can be preserved under lead reduction.

## Included Files
- `pclr_partial_lead_eval.py`
- `simclr_partial_lead_evaluation.py`
- `supervised_partial_lead_evaluation.py`
- `hybrid_partial_lead_evaluation.py`

## Motivation
Full 12-lead ECG acquisition is informative but not always practical. Reduced-lead systems are relevant for mobile devices, wearable sensors, telehealth workflows, and lower-cost screening setups. This folder tests how different training strategies respond when lead information is restricted.

## Lead Configurations
The evaluation pipeline includes multiple lead subsets, such as:
- full 12-lead input,
- limb-only leads,
- chest-only leads,
- selected 3-lead subsets,
- single-lead Lead II,
- and single-lead V5.

## What the Scripts Do
Each script loads a trained model, masks out unavailable leads, and evaluates predictive performance using only the selected subset. Metrics may include:
- accuracy,
- balanced accuracy,
- and macro F1 score.

## Script Descriptions

### `pclr_partial_lead_eval.py`
Evaluates a PCLR-based model under reduced-lead conditions. It is useful for testing whether patient-aware self-supervised representations remain informative even when many channels are removed.

### `simclr_partial_lead_evaluation.py`
Evaluates the SimCLR-based model under the same reduced-lead settings.

### `supervised_partial_lead_evaluation.py`
Measures how the fully supervised baseline behaves when fewer leads are retained.

### `hybrid_partial_lead_evaluation.py`
Tests the hybrid model under partial-lead constraints, allowing comparison against both pure supervised and pure contrastive strategies.

## Expected Outcome
These experiments help answer questions such as:
- Which lead subsets preserve the most performance?
- Do self-supervised representations degrade more gracefully than supervised ones?
- Are some single-lead or low-lead configurations surprisingly competitive?

## Notes
For fair comparison:
- use the same preprocessing pipeline,
- keep test conditions consistent,
- and compare all model families on the same lead subsets.
