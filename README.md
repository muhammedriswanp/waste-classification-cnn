# Waste Classification CNN

## Dataset
- 4650 images, 6 classes: battery, glass, metal, organic, paper, plastic
- Train: 3720 | Val: 930

## Results (1st Run)

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 01/10 | 44.54% | 63.87% | 0.9306 |
| 02/10 | 53.82% | 62.58% | 0.9585 |
| 03/10 | 56.18% | 63.33% | 0.9240 |
| 04/10 | 57.20% | 68.28% | 0.8037 |
| 05/10 | 59.62% | 68.06% | 0.8384 |
| 06/10 | 59.17% | 66.24% | 0.9060 |
| 07/10 | 59.70% | 69.68% | 0.7871 |
| 08/10 | 61.56% | 66.45% | 0.8456 |
| **09/10** | **60.67%** | **70.97%** | **0.7733** |
| 10/10 | 61.80% | 70.86% | 0.7795 |

**Best:** Val Acc 70.97% | Val Loss 0.7733 (Epoch 09)

### Classification Report

| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| battery  | 0.73      | 0.78   | 0.76     |
| glass    | 0.70      | 0.53   | 0.61     |
| metal    | 0.41      | 0.40   | 0.41     |
| organic  | 0.83      | 0.75   | 0.79     |
| paper    | 0.68      | 0.75   | 0.71     |
| plastic  | 0.47      | 0.58   | 0.52     |

**Accuracy:** 63% | **Macro Avg:** 0.64 / 0.63 / 0.63

## Quick Start
```bash
pip install -r requirements.txt
python train.py
```