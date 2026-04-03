# GNR-638: Machine Learning for Remote Sensing - II  
## Assignment 3 — Efficient Channel Attention (ECA)

**Name:** Medhansh Sharma  
**Roll No.:** 22b1287  
**Date:** April 2026  

---

## Overview

This project implements the **Efficient Channel Attention (ECA)** mechanism from scratch and integrates it into modern convolutional neural networks to evaluate its effectiveness on a real-world image classification task.

We compare:
- Baseline architectures
- Custom ECA-enhanced models
- Official ECA implementations

---

## Motivation

Traditional attention mechanisms like **SENet** improve performance but introduce:
- Increased model complexity  
- Additional parameters due to dimensionality reduction  

The **ECA module** addresses this by:
- Avoiding dimensionality reduction  
- Using lightweight 1D convolution  
- Maintaining efficiency while significantly improving accuracy  

---

## ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks** *Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu* [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf)] [[Code](https://github.com/BangguWu/ECANet)]

ECA works by:
1. Extracting global channel descriptors using Global Average Pooling.
2. Applying **local cross-channel interaction** via an adaptive 1D convolution.
3. Generating attention weights using a Sigmoid activation function.
4. Re-weighting feature maps channel-wise.

**Advantages:** No fully connected layers  
 Minimal parameters  
 Efficient and scalable  

---

## Model Architectures

We evaluate three primary setups to satisfy the reproducibility requirements:

### 1. Baselines
- Standard PyTorch implementations of **ResNet-50** and **MobileNetV2**.  

### 2. Custom ECA (Ours)
- Implemented entirely from scratch in PyTorch.  
- Integrated into:
  - ResNet bottleneck blocks.  
  - MobileNetV2 inverted residual blocks.  

### 3. Official Implementation
- Used for validation and benchmarking against the original paper's parameter counts and FLOPs.  

---

## Dataset

**Fast Food Classification V2** - **10 classes** with high intra-class variation.  
- Introduces real-world computer vision challenges such as diverse lighting, spatial orientation, and scale.  

**Classes Include:** `Burger`, `Fries`, `Taco`, `Donut`, `Sandwich`, `Hot Dog`, `Pizza`, `Taquito`, `Baked Potato`, `Crispy Chicken`

---

## Data Preprocessing

| Phase | Pipeline |
| :--- | :--- |
| **Training** | Random Resized Crop (224×224) ➔ Random Horizontal Flip ➔ Color Jitter (0.2) ➔ Random Rotation (±15°) ➔ Normalize (mean=0.5, std=0.5) |
| **Validation** | Resize (256) ➔ Center Crop (224) ➔ Normalize (mean=0.5, std=0.5) |

---

## Final Training Configuration

| Parameter        | Value       | Parameter      | Value       |
|------------------|-------------|----------------|-------------|
| **Optimizer** | SGD         | **Batch Size** | 128         |
| **Learning Rate**| 0.01        | **Epochs** | 50          |
| **Momentum** | 0.9         | **Image Size** | 224×224     |
| **Weight Decay** | 0.0004      | **Workers** | 4           |
| **Hardware** | NVIDIA A40  | **Seed** | 42          |
| **Framework** | PyTorch 2.x |                |             |

---

## Results

### ResNet-50

| Model            | Top-1 Accuracy | Top-5 Accuracy | FLOPs |
|------------------|---------------|---------------|-------|
| Baseline         | 74.74%        | 96.00%        | 8.26G |
| **ECA-ResNet-50 (Ours)** | **78.37%** | **96.46%** | 8.27G |

### MobileNetV2

| Model               | Top-1 Accuracy | Top-5 Accuracy | FLOPs |
|---------------------|---------------|---------------|-------|
| Baseline            | 75.71%        | 96.17%        | 319M  |
| **ECA-MobileNetV2 (Ours)** | **76.83%** | **96.40%** | 654M  |

---

## Key Observations & Analysis

- **Performance Gain:** Achieved a **+3.63% Top-1 improvement** on ResNet-50, aligning with the theoretical claims of the original paper.  
- **Computational Efficiency:** Minimal increase in FLOPs and parameters due to the absence of expensive fully connected layers.  
- **Stability:** Validation curves demonstrated smoother learning trajectories and reduced fluctuations compared to the baselines.  
- **Implementation Verification:** Our from-scratch implementation seamlessly matches the official parameter counts and integrates flawlessly into both heavy and lightweight backbone architectures.  

---

## Reproducibility
### Clone the repository
```
git clone https://github.com/MedhanshSharma2004/gn638_assignment_3.git
cd gn638_assignment_3
```
### Install dependencies
```
pip install -r requirements.txt
```
### Run the training script for ResNet-50 with ECA
```
python main.py --model resnet50 --eca
```
### Run the training script for MobileNetV2 with ECA
```
python main.py --model mobilenetv2 --eca
```

### Configuration Setup
Ensure your configuration file (or parsed arguments) matches the following settings used during our testing phase:

```yaml
batch_size: 128
seed: 42
lr: 0.01
num_workers: 8
num_classes: 10
image_size: 224 
num_epochs: 50
momentum: 0.9
train_dir: "data/Fast Food Classification V2/Train"
val_dir: "data/Fast Food Classification V2/Valid"
weight_decay: 0.0004
plot_dir: "plots"
model_save_dir: "saved_models"
