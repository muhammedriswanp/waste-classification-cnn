# Waste Classification CNN
## Model Performance
- Best Validation Accuracy: 76.67%
- Best Validation Loss: 0.7750
- Training Epochs: 16 (early stopping)
## Dataset
- 6 classes: battery, glass, metal, organic, paper, plastic
- Total images: 4650
- Train/Val split: 80/20
## Model Architecture
- 3 Conv blocks (32→64→128 channels)
- FC: 32768→256→6
## Quick Start
1. pip install -r requirements.txt
2. python train.py
3. python inference.py --image path/to/image.jpg