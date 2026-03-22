# VGG (2014) – Paper Replication

## Overview

This implementation is based on the architecture described in the paper:

**Karen Simonyan, Andrew Zisserman — _Very Deep Convolutional Networks for Large-Scale Image Recognition_ (2014).**

VGG is one of the most influential convolutional neural networks in deep learning history and demonstrated that increasing depth using small, uniform convolutional filters (3×3) can significantly improve performance. The architecture introduced a simple and scalable design that became foundational for modern deep learning models.

This repository provides **two ways to interact with the VGG replication**:

1. **Modular Python Implementation** – designed for reproducible experiments, training, and evaluation.
2. **Interactive Notebook Version** – designed for easier exploration, visualization, and experimentation.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/HdhwivHel/VGG-Paper-Replication
cd VGG-Paper-Replication
```

````

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```
Python ≥ 3.10
```

---

## 1. Modular Implementation

The modular version is intended for **structured experiments and reproducibility**.
It separates the model architecture, dataset handling, configuration, and training scripts.

### Train the model

```bash
python training/train.py
```

Training parameters such as **epochs, batch size, and learning rate** can be modified in:

```
configs/config.yaml
```

### Evaluate the trained model

```bash
python training/test.py
```

---

## 2. Notebook Implementation

The notebook (`main.ipynb`) provides a **step-by-step implementation of VGG** that is easier to follow and experiment with.

Colab Link:
https://colab.research.google.com/github/HdhwivHel/VGG-Paper-Replication/blob/main/main.ipynb

The notebook includes:

- Model construction
- Training loop
- Evaluation
- Exploration of predictions on validation images

---

# Differences from Original VGG Paper

## 1. Architecture Differences

| Component                | Original Paper                     | This Implementation    | Reason                                        |
| ------------------------ | ---------------------------------- | ---------------------- | --------------------------------------------- |
| Output Layer             | `Linear(4096 → 1000)` for ImageNet | `Linear(4096 → 100)`   | Training performed on **ImageNet-100 subset** |
| Framework Implementation | Custom Caffe implementation        | PyTorch implementation | Modern deep learning framework                |

All core architectural components, including **3×3 convolutions, ReLU activations, max pooling, and dropout**, are preserved.

---

## 2. Training Differences

| Component              | Original Paper                             | This Implementation       | Reason                              |
| ---------------------- | ------------------------------------------ | ------------------------- | ----------------------------------- |
| Epochs                 | ~74 epochs                                 | **50 epochs**             | Smaller dataset (ImageNet-100)      |
| Learning Rate Schedule | LR reduced when validation error plateaued | Constant LR               | Easier Setup                        |
| Hardware Setup         | Multi-GPU training                         | Single GPU                | Modern hardware capability          |

---

## 3. Input Preprocessing Differences

| Component           | Original Paper                       | This Implementation             | Reason                        |
| ------------------- | ------------------------------------ | ------------------------------- | ----------------------------- |
| Training Crops      | Random 224 crops from resized images | `Resize(256) → RandomCrop(224)` | Standard torchvision pipeline |
| Horizontal Flipping | Used                                 | Same                            | Data augmentation             |
| Color Augmentation  | Scale jittering                      | Not implemented                 | Simplified pipeline           |
| Tensor Conversion   | Custom preprocessing                 | `ToTensor()`                    | PyTorch pipeline              |

---

## 4. Model Architecture (Identical Components)

Despite the implementation differences above, the following architectural properties remain identical to the original VGG design.

| Property       | Value                        |
| -------------- | ---------------------------- |
| Input size     | 224 × 224                    |
| Conv Layers    | 3×3 kernels, stride 1, pad 1 |
| Pooling        | 2×2 max pooling, stride 2    |
| Depth          | 16 / 19 layers               |
| FC1            | 4096 neurons                 |
| FC2            | 4096 neurons                 |
| Output classes | 100                          |

Dropout is applied to the first two fully connected layers with:

```
Dropout probability = 0.5
```

---

# Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | SGD              |
| Learning Rate | 0.01             |
| Momentum      | 0.9              |
| Weight Decay  | 0.0005           |
| Batch Size    | 32–64            |
| Epochs        | 50               |
| Loss Function | CrossEntropyLoss |

---

# Dataset

Training is performed on **ImageNet-100**, a subset of the ImageNet ILSVRC 2012 dataset.

Dataset statistics:

```
Number of classes: 100
Training images: ~130k
Validation images: ~5k
```

Images are resized and cropped to match the **224×224 input resolution expected by VGG**.

---

# Summary

This implementation preserves the **core VGG architecture**, including **stacked 3×3 convolutions, deep hierarchical feature extraction, and dropout regularization**, while adopting modern deep learning practices such as:

- PyTorch implementation
- modular training pipeline
- configurable experiments
- reduced dataset size

The goal of this project is to **faithfully reproduce the VGG architecture and training setup** while making the model accessible for experimentation on modern hardware.

---
