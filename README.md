# 🌦️ Weather Image Classification using Custom CNN (PyTorch)

A complete PyTorch pipeline for multi-class weather image classification. This project implements a custom Convolutional Neural Network (CNN) from scratch and compares architecture variants with and without Batch Normalization and Dropout to analyze their impact on training stability and generalization.

---

## 📊 Dataset Overview

The model is trained on a Multi-class Weather Dataset containing images across four distinct categories.

- **Total Images:** 1,125
- **Classes:** Cloudy, Rain, Shine, Sunrise
- **Class Distribution:**
  - Cloudy: 300 images
  - Rain: 215 images
  - Shine: 253 images
  - Sunrise: 357 images
- **Data Split:** 
  - Training: 787 images (70%)
  - Validation: 168 images (15%)
  - Test: 170 images (15%)

<p align="center">
  <img src="Different Sample .png" alt="Sample Images" width="900">
</p>

---

## ⚙️ Methodology & Pipeline

1. **Data Preprocessing & Augmentation:**
   - **Training Transforms:** Resize (224x224), Random Horizontal Flip (p=0.5), Random Rotation (±15°), Color Jitter (brightness/contrast), Normalization (ImageNet stats).
   - **Validation/Test Transforms:** Resize (224x224), Normalization only.
2. **Custom CNN Architecture:**
   - 4 Convolutional Blocks (32 → 64 → 128 → 256 filters).
   - Max Pooling (2x2) after each block.
   - Batch Normalization and Dropout (0.5) applied for regularization.
   - Fully Connected Layers: 50,176 → 512 → 256 → 4.
   - Total Trainable Parameters: **26,212,356**.
3. **Training Strategy:**
   - **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
   - **Loss Function:** CrossEntropyLoss
   - **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
   - **Epochs:** 30
4. **Model Comparison:** 
   - Trained three variants (15 epochs each) to evaluate the impact of regularization:
     - With BatchNorm + Dropout
     - Without BatchNorm (with Dropout)
     - Without Dropout (with BatchNorm)

---

## 🤖 Model Performance & Results

### Final Test Set Evaluation (Best Model: With BN + Dropout)

- **Test Accuracy:** 92.35%
- **Test Loss:** 0.2856
- **Best Validation Accuracy:** 95.24%

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Cloudy | 0.95 | 0.86 | 0.90 | 43 |
| Rain | 0.96 | 0.87 | 0.91 | 30 |
| Shine | 0.78 | 0.97 | 0.87 | 37 |
| Sunrise | 1.00 | 0.97 | 0.98 | 60 |
| **Accuracy** | | | **0.92** | **170** |
| **Macro Avg** | 0.92 | 0.92 | 0.92 | 170 |
| **Weighted Avg** | 0.93 | 0.92 | 0.93 | 170 |

### Per-Class Accuracy
- **Sunrise:** 96.67% (Best performing)
- **Shine:** 97.30% (Best performing)
- **Rain:** 86.67%
- **Cloudy:** 86.05% (Lowest performing)

### Misclassification Analysis
- Total misclassifications: **13 out of 170** images.
- The confusion matrix revealed that the model struggles most with distinguishing between `Cloudy` and `Rain` classes, likely due to visual similarities in overcast conditions.

---

## 📈 Training & Evaluation Visualizations

### Training and Validation Curves
The model showed rapid convergence. With the `ReduceLROnPlateau` scheduler, the learning rate dropped from 0.001 to 0.000063 by epoch 30, allowing the model to fine-tune and achieve high validation accuracy without severe overfitting.

<p align="center">
  <img src="training and validation (Accuracy and loss).png" alt="Training Curves" width="900">
</p>

### Confusion Matrix
The confusion matrix highlights the model's strong performance on the `Sunrise` class and identifies the specific misclassifications between `Cloudy` and `Rain`.

<p align="center">
  <img src="Confusion Matrix on test set.png" alt="Confusion Matrix" width="700">
</p>

---

## 🏗️ Project Structure

```text
Weather-Classification/
│
├── data/                           # Dataset folder (Multi-class Weather Dataset)
├── model_with_bn_dropout_best.pth  # Best model weights
├── weather_cnn_final.pth           # Final saved model
├── training_history.json           # Training metrics history
├── classification_report.txt       # Detailed test set report
├── Weather_CNN_Complete_Code.py    # Main training and evaluation script
└── README.md                       # Project documentation
