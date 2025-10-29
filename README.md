# Chili Leaf Disease Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://tensorflow.org)
[![DOI](https://img.shields.io/badge/DOI-10.17632/ymt8k9bjkn.2-brightgreen.svg)](https://doi.org/10.17632/ymt8k9bjkn.2)

Author: [Bagavati Narayanan](https://github.com/baggie11)

A deep learning-based computer vision system for automated detection and classification of chili leaf diseases using the Chili Leaf Diseases Dataset from the Krishna River Basin.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)

## Overview

This project implements a comprehensive solution for chili leaf disease classification using deep learning. The system can accurately identify various diseases affecting chili plants, enabling early detection and intervention for farmers.

**Key Highlights:**
- Multi-class classification of chili leaf diseases
- Transfer learning with state-of-the-art architectures
- Comprehensive data preprocessing and augmentation pipeline
- Model interpretability and visualization tools

## Dataset

### Source
**Chili Leaf Diseases Dataset from the Krishna River Basin**

**Citation:**
```bibtex
@dataset{malghan2025chili,
  author = {Malghan, Rashmi Laxmikant and K, Lingaraj and M C, Karthik Rao and Garg, Lalit},
  title = {Image Dataset on Chili Leaf Diseases in the Krishna River Basin of the Deccan Plateau, India},
  year = {2025},
  publisher = {Mendeley Data},
  version = {V2},
  doi = {10.17632/ymt8k9bjkn.2},
  url = {https://data.mendeley.com/datasets/ymt8k9bjkn/2}
}

# Methodology

## 1. Dataset Preparation

- **Image Dataset**: Collected and organized into class-labeled folders. Images are of size 224×224.
- **Class Imbalance**: The dataset contains minority and majority classes.
- **Transforms**:
  - `base_transform`: Resizes images and converts them to tensors (used for majority classes or validation).
  - `augment_transform`: Applies data augmentation for minority classes, including:
    - Random resized crop
    - Horizontal flip
    - Random rotation
    - Color jitter (brightness, contrast, saturation, hue)
    - Conversion to tensor

---

## 2. BalancedAugmentedDataset

- **Purpose**: To handle class imbalance by oversampling minority classes and applying augmentation.
- **Implementation**:
  1. Compute class counts.
  2. Identify minority classes.
  3. Oversample minority classes by duplicating samples.
  4. Apply augmentation (`augment_transform`) only to oversampled minority class images.
  5. Shuffle dataset to mix augmented and original samples.
- **Training vs Validation**:
  - Training: Augmented and balanced.
  - Validation: Original class distribution, no augmentation.

---

## 3. Stratified Train/Validation Split

- **Reason**: To preserve class proportions in both splits.
- **Method**: 
  - Use `train_test_split` from `sklearn` with `stratify` parameter on labels.
  - Typically, 80% of data for training and 20% for validation.
- Ensures minority classes are represented in both splits.

---

## 4. CNN Architecture (SimpleCNN)

- **Feature Extractor**:
  - 3 Convolutional layers (with ReLU) and MaxPooling.
  - Extracts hierarchical features: edges → textures → shapes.
- **Flatten Layer**: Converts 3D feature maps into 1D vector for fully connected layers.
- **Classifier**:
  - Fully connected layer reducing feature dimension.
  - ReLU activation + Dropout (0.5) to prevent overfitting.
  - Output layer: `num_classes` logits for classification.
- **Dynamic Flattening**: Automatically calculates the size of flattened features from the input image.

---

## 5. Model Training

- **Device**: GPU if available, else CPU.
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Adam with learning rate 1e-3.
- **Epochs**: 10 (can be tuned based on convergence).
- **Training Loop**:
  1. Forward pass through the model.
  2. Compute loss.
  3. Backpropagation and optimizer step.
  4. Track training loss and accuracy.
- **Validation Loop**:
  1. Forward pass on validation set.
  2. Compute validation loss and accuracy.
  3. Collect predictions for confusion matrix.

---

## 6. Evaluation & Visualization

- **Metrics**:
  - Training and validation loss over epochs.
  - Training and validation accuracy over epochs.
- **Visualization**:
  - Plot loss curves for training and validation.
  - Plot accuracy curves for training and validation.
  - Confusion matrix on validation set to analyze per-class performance.

---

## 7. Key Points

- Balanced augmentation ensures **minority classes are sufficiently represented**.
- Augmentation is applied **only to minority class oversamples**, avoiding overfitting majority classes.
- Stratified splitting preserves **class distribution** in both training and validation sets.
- SimpleCNN is **lightweight but sufficient** for 224×224 images and works with this dataset size.

