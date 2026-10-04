# Mechanical Parts Semantic Segmentation

### MLP Project 2026 T2

A deep learning project for **multi-class semantic segmentation** of mechanical parts in cluttered industrial scenes.

The objective is to classify every pixel of a 384×384 RGB image into one of seven classes: background or one of six mechanical components — **hex nuts, washers, bolts, ball bearings, springs, and O-rings**.

The project implements a **U-Net segmentation model from scratch**, without using pretrained weights, and trains it using a combination of **Cross-Entropy Loss and Dice Loss**.

---

## Project Overview

Automated visual inspection and robotic sorting systems need to identify individual components even when objects are:

* Arbitrarily oriented
* Partially overlapping
* Similar in appearance
* Made from similar materials
* Surrounded by other components

This competition presents images of mechanical parts scattered across a workbench and requires pixel-level segmentation of every component.

The project uses a custom U-Net architecture to produce a seven-class segmentation mask for each image.

### Competition Task

Given:

* **2,000 labelled training images**
* **500 unseen test images**
* Image resolution: **384 × 384**
* RGB input images

The model must predict the class of every pixel.

---

## Classes

The segmentation task contains seven labels including the background:

| Class ID | Class        |
| -------: | ------------ |
|        0 | Background   |
|        1 | Hex Nut      |
|        2 | Washer       |
|        3 | Bolt         |
|        4 | Ball Bearing |
|        5 | Spring       |
|        6 | O-Ring       |

Each training mask stores the class ID directly as the pixel value.

---

# Evaluation Metric

The competition is evaluated using the **Dice coefficient**.

For predicted pixel set `X` and ground-truth pixel set `Y`:

```text
Dice = 2 × |X ∩ Y| / (|X| + |Y|)
```

The score ranges from:

```text
0 → No overlap
1 → Perfect overlap
```

Higher Dice scores indicate better segmentation performance.

The competition score is calculated as the mean Dice coefficient across all image/class pairs.

Because the official competition metric is Dice, the project uses **validation Dice as the primary model-selection metric**.

---

# Dataset

The dataset contains:

```text
2,000 labelled training images
500 unlabelled test images
384 × 384 RGB images
```

### Dataset Structure

```text
dataset/
│
├── train/
│   ├── images/
│   └── masks/
│
├── test/
│   └── images/
│
├── train.csv
├── sample_submission.csv
├── metadata.csv
└── baseline_starter.py
```

### Training Images

The training images are RGB PNG files with a resolution of:

```text
384 × 384 × 3
```

### Segmentation Masks

The corresponding masks contain integer class IDs from `0` to `6`.

The masks are loaded directly as arrays:

```python
label = np.array(Image.open(mask_path))
```

The masks are **not converted to RGB**, since the underlying pixel values represent the segmentation labels.

---

# Exploratory Data Analysis

Before model development, the dataset was investigated to understand:

* Image dimensions
* Image data types
* Pixel value ranges
* Mask dimensions
* Unique segmentation labels
* Foreground/background distribution
* Class distribution
* Dataset statistics

### Foreground Coverage

The analysis showed that approximately:

```text
24.67% → Average foreground coverage
75.33% → Average background coverage
```

Foreground coverage ranged approximately from:

```text
15.84% → 40.14%
```

This indicates a significant amount of background in each image and introduces class imbalance into the segmentation task.

This motivated the use of **Dice-based optimization**, which focuses directly on segmentation overlap.

---

# RLE Verification

The competition requires segmentation masks to be submitted using **Run-Length Encoding (RLE)**.

The project implements both:

```text
rle_encode()
rle_decode()
```

The encoding uses column-major / Fortran ordering, matching the competition specification.

A round-trip verification was performed:

```python
assert np.array_equal(
    rle_decode(rle_encode(mask)),
    mask.astype(np.uint8)
)
```

This ensures that the generated RLE can correctly reconstruct the original binary mask.

---

# Data Preprocessing

An **85/15 train-validation split** was created to evaluate model performance during training.

```text
85% → Training
15% → Validation
```

The split was performed with reproducibility in mind using a fixed random seed.

Dataset-specific channel-wise mean and standard deviation were calculated from the training images and used for normalization.

---

# Data Augmentation

Training images were augmented using the following transformations:

### Spatial Augmentation

* Horizontal Flip
* Vertical Flip
* Random 90° Rotation
* Random Rotation up to ±20°

### Appearance Augmentation

* Random Brightness Adjustment
* Random Contrast Adjustment

The masks use nearest-neighbor interpolation during geometric transformations to preserve their discrete class labels.

Validation images receive only normalization without random augmentation.

---

# PyTorch Dataset

A custom PyTorch `Dataset` was implemented to:

1. Load RGB images.
2. Load corresponding segmentation masks.
3. Apply image/mask transformations.
4. Convert them into PyTorch tensors.

Efficient `DataLoader` objects were then created for training, validation, and test inference.

### Training Configuration

```text
Batch Size : 8
Train Split: 85%
Validation : 15%
```

---

# Model Architecture

## U-Net

A custom **U-Net** architecture was implemented from scratch.

U-Net is well suited for semantic segmentation because it combines:

* Deep semantic feature extraction
* High-resolution spatial information
* Encoder-decoder architecture
* Skip connections

The encoder progressively reduces spatial resolution while increasing feature depth.

The decoder then reconstructs the segmentation map while using skip connections to recover fine-grained spatial information.

---

## Architecture

```text
Input
384 × 384 × 3
      │
      ▼
Encoder 1
64 channels
      │
      ▼
Encoder 2
128 channels
      │
      ▼
Encoder 3
256 channels
      │
      ▼
Encoder 4
512 channels
      │
      ▼
Bottleneck
1024 channels
      │
      ▼
Decoder 4
512 channels
      │
      ▼
Decoder 3
256 channels
      │
      ▼
Decoder 2
128 channels
      │
      ▼
Decoder 1
64 channels
      │
      ▼
1 × 1 Convolution
      │
      ▼
7 Classes
384 × 384 × 7
```

The network contains:

* Four encoder blocks
* A 1024-channel bottleneck
* Four decoder blocks
* Skip connections
* Dropout (`p = 0.30`)
* Final 1×1 convolution producing seven classes

---

# No Pretrained Weights

The competition prohibits the use of pretrained model weights.

Therefore, the U-Net model was trained **entirely from scratch** using the competition training data.

Weights were initialized using:

```python
Kaiming Normal Initialization
```

for convolutional and transposed-convolutional layers.

Batch normalization parameters were initialized separately.

---

# Loss Function

The model uses a combination of **Cross-Entropy Loss and Dice Loss**.

```text
Total Loss = Cross Entropy Loss + Dice Loss
```

### Cross-Entropy Loss

Cross-Entropy provides stable pixel-level classification supervision.

### Dice Loss

Dice Loss directly encourages overlap between predicted and ground-truth segmentation masks:

```text
Dice Loss = 1 − Dice Score
```

Using both losses allows the model to benefit from:

* Pixel-level classification
* Overlap-aware optimization

This is particularly useful because the dataset contains a substantial amount of background relative to foreground pixels.

---

# Optimization

The model was trained using the **AdamW** optimizer.

```text
Optimizer       : AdamW
Learning Rate   : 0.001
Weight Decay    : 0.0001
Epochs          : 30
Batch Size      : 8
```

---

# Learning Rate Scheduling

A `ReduceLROnPlateau` scheduler was used to adjust the learning rate when validation Dice stopped improving.

Configuration:

```text
Mode      : max
Factor    : 0.5
Patience  : 3
Minimum LR: 1e-6
```

The scheduler monitors validation Dice because higher Dice indicates better segmentation performance.

---

# Mixed Precision Training

Training uses **Automatic Mixed Precision (AMP)** on NVIDIA GPUs.

AMP provides:

* Lower GPU memory usage
* Faster computation
* Efficient training on compatible hardware
* Gradient scaling for numerical stability

This allowed the U-Net to be trained efficiently while processing relatively large 384×384 images.

---

# Early Stopping & Model Checkpointing

An early stopping mechanism was implemented with:

```text
Patience = 8 epochs
```

The model checkpoint was updated whenever validation Dice improved.

The best-performing weights were saved as:

```text
best_unet_model.pth
```

This ensures that the final inference model corresponds to the best validation performance rather than simply the final training epoch.

---

# Training Results

The model showed rapid improvement during training.

| Epoch | Training Loss | Validation Loss | Validation Dice |
| ----: | ------------: | --------------: | --------------: |
|     1 |        1.3402 |          1.0371 |          0.5497 |
|     5 |        0.2461 |          0.1869 |          0.9280 |
|    10 |        0.1195 |          0.0964 |          0.9655 |
|    15 |        0.0916 |          0.0801 |          0.9709 |
|    20 |        0.0821 |          0.0713 |          0.9747 |
|    25 |        0.0720 |          0.0592 |          0.9791 |
|    30 |        0.0608 |          0.0509 |      **0.9817** |

### Best Validation Performance

```text
Best Validation Dice: 0.9817
Epoch: 30
```

The validation Dice improved from approximately **0.55 in the first epoch to 0.98 by the final epoch**, indicating strong learning of the segmentation task.

---

# Inference

After training, the checkpoint with the highest validation Dice was loaded for test inference.

The model produces a seven-class segmentation mask:

```text
384 × 384
```

Each pixel contains a predicted class ID.

The predicted mask is then separated into six foreground binary masks:

```text
hex_nut
washer
bolt
ball_bearing
spring
o_ring
```

The background class is not submitted separately.

---

# Submission Generation

The competition requires **six rows per test image**, one for each foreground class.

For:

```text
500 test images
×
6 foreground classes
```

the submission contains:

```text
3,000 rows
```

with the following format:

```text
ImageId_ClassId,EncodedPixels
```

Example:

```text
img_00007.png_hex_nut,...
img_00007.png_washer,...
img_00007.png_bolt,...
img_00007.png_ball_bearing,...
img_00007.png_spring,...
img_00007.png_o_ring,...
```

Each predicted binary mask is converted to RLE before being added to the submission file.

If a class is not predicted in an image, the submission uses:

```text
1 1
```

as the encoded value, following the notebook's submission-generation logic.

---

# End-to-End Pipeline

```text
                RGB Images
                    │
                    ▼
             Dataset Analysis
                    │
                    ▼
           Train / Validation Split
                 85 / 15
                    │
                    ▼
           Data Augmentation
                    │
                    ▼
            Normalization
                    │
                    ▼
             Custom Dataset
                    │
                    ▼
               U-Net
          (trained from scratch)
                    │
                    ▼
        Cross Entropy + Dice Loss
                    │
                    ▼
                AdamW
                    │
                    ▼
          AMP + LR Scheduling
                    │
                    ▼
          Validation Dice Score
                    │
                    ▼
        Best Model Checkpoint
                    │
                    ▼
             Test Inference
                    │
                    ▼
          Six Binary Class Masks
                    │
                    ▼
                RLE Encoding
                    │
                    ▼
             submission.csv
```

---

# Technologies Used

### Programming

* Python

### Deep Learning

* PyTorch
* TorchVision

### Image Processing

* OpenCV
* Pillow
* Albumentations

### Data Analysis

* NumPy
* Pandas
* Matplotlib
* Seaborn

### Environment

* Kaggle Notebook
* NVIDIA GPU
* Automatic Mixed Precision

---

# Key Learnings

This project provided practical experience with:

* Semantic segmentation
* Pixel-level multi-class classification
* U-Net architecture
* Encoder-decoder networks
* Skip connections
* Custom PyTorch datasets
* Image and mask augmentation
* Class imbalance
* Dice coefficient
* Dice Loss
* Cross-Entropy Loss
* Combined segmentation losses
* AdamW optimization
* Learning-rate scheduling
* Early stopping
* Mixed precision training
* Model checkpointing
* Run-Length Encoding
* Competition-style submission generation

---

# Project Structure

```text
Mechanical-Parts-Semantic-Segmentation/
│
├── 24f3004946-notebook-26t2.ipynb
├── README.md
├── best_unet_model.pth
└── submission.csv
```

The original competition dataset is not included in the repository.

---

# Author

**Soumyadip Hazari**

BS Degree in Data Science and Applications
Indian Institute of Technology Madras

GitHub: [SoumyadipHazari](https://github.com/SoumyadipHazari)

---
