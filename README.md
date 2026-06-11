# 🌸 FloriVision — Two-Stage Computer Vision Pipeline

> **YOLOv8 Object Detection + ResNet-18 Transfer Learning**  
> Track A — Computer Vision Capstone | AI & GPU Computing Internship 2026  
> Presidency School of AI and Advanced Computing, Presidency University

---

## 📊 Results

| Model | Validation Accuracy | Parameters | Training Time (H200) |
|---|---|---|---|
| ResNet-18 (Transfer Learning) | **87.5%** | 11.2M | ~3 min |
| YOLOv8n-cls (Transfer Learning) | **95.1%** | 2.7M | ~1 min |

---

## 🎯 Project Overview

FloriVision is a two-stage Computer Vision pipeline that combines **object detection** and **image classification** to identify five flower species from any image or live camera feed.

```
Input Image / Camera
        ↓
Stage 1 — YOLOv8 (Detection)
        → Detects and localises the flower region
        → Draws bounding box around the flower
        ↓
Stage 2 — ResNet-18 (Classification)
        → Classifies the flower type from the detected region
        → Returns: "Rose — 97% confident"
        ↓
Output: Labelled image with bounding box + flower name
```

**Flower Classes:** Daisy 🌼 | Dandelion 🌻 | Roses 🌹 | Sunflowers 🌞 | Tulips 🌷

---

## 🗂️ Project Structure

```
Flower-Classification/
│
├── flower_classifier.py      # Main training script (CPU version)
├── flower_classifier_h200.py # H200 GPU optimised training
├── yolo_addon.py             # YOLOv8 addition + two-stage pipeline (CPU)
├── yolo_addon_h200.py        # YOLOv8 H200 GPU version
├── plot_predictions.py       # Improved predictions visualisation
├── live_detector.py          # Real-time webcam flower detection
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Kishore0760/Flower-Classification.git
cd Flower-Classification
```

### 2. Install dependencies
```bash
pip install torch torchvision matplotlib scikit-learn pillow tqdm ultralytics opencv-python
```

### 3. Train the model
```bash
python flower_classifier.py
```
The dataset (~220MB) auto-downloads on first run. No setup needed.

### 4. Add YOLOv8 detection
```bash
python yolo_addon.py
```

### 5. Run live camera detection
```bash
python live_detector.py
```

---

## 🖥️ H200 GPU Version

For running on NVIDIA H200 via JupyterHub:

```bash
# Upload to JupyterHub, then in terminal:
pip install ultralytics
python flower_classifier_h200.py   # ~3-5 min on H200
python yolo_addon_h200.py          # ~1-2 min on H200
```

**H200 improvements over CPU:**

| Setting | CPU | H200 |
|---|---|---|
| Batch size | 32 | 128 |
| Epochs | 10 | 20 |
| Training time | ~40 min | ~3 min |
| Expected accuracy | 87.5% | 92–96% |
| Fine-tuning | Frozen backbone | Full network |
| Mixed precision | No | Yes (FP16) |

---

## 📁 Dataset

**Google Flowers Dataset**
- 3,670 images across 5 classes
- Source: [TensorFlow Example Images](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
- Split: 80% train / 20% validation
- Auto-downloaded by the script — no manual setup needed

| Class | Images |
|---|---|
| Daisy | 633 |
| Dandelion | 898 |
| Roses | 741 |
| Sunflowers | 671 |
| Tulips | 733 |
| **Total** | **3,670** |

---

## 🔬 Methodology

### Transfer Learning
Both models use ImageNet pretrained weights, fine-tuned for flower classification:
- Pre-trained backbone frozen → only final classification layer trained
- Dramatically reduces training data requirement (3,670 vs millions of images)
- ResNet-18: replaced `Linear(512, 1000)` → `Linear(512, 5)`
- YOLOv8: classification head replaced for 5-class output

### Data Augmentation (Training only)
```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Two-Stage Pipeline
- **Stage 1** — Pretrained YOLOv8 (COCO, 80 classes) detects the object region
- **Stage 2** — Fine-tuned ResNet-18 classifies the flower type from the cropped region

---

## 📈 Output Files

After running the scripts you get:

```
resnet18_best.pth           ← saved model weights
resnet18_curves.png         ← training accuracy + loss curves
resnet18_confusion.png      ← confusion matrix
sample_predictions.png      ← 3x3 prediction grid
sample_predictions_v2.png   ← improved dark-theme grid
model_comparison.png        ← ResNet-18 vs ResNet-50 vs EfficientNet
yolo_vs_resnet.png          ← YOLOv8 vs ResNet-18 comparison
two_stage_pipeline.png      ← detection + classification demo
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.14 | Core language |
| PyTorch | 2.12 | Deep learning framework |
| torchvision | 0.27 | Pretrained models + transforms |
| Ultralytics | Latest | YOLOv8 detection + classification |
| scikit-learn | 1.9 | Confusion matrix + metrics |
| OpenCV | 4.13 | Live camera feed |
| matplotlib | 3.10 | Visualisations |
| NVIDIA H200 | Hopper | 141GB HBM3e GPU (training) |

---

## 📖 References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
3. Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*.
4. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*.

---

## 👥 Team

**AI & GPU Computing Summer Internship 2026**  
Presidency School of AI and Advanced Computing  
Presidency University, Bangalore

---

## 📄 License

This project is submitted as part of the AI & GPU Computing Internship Program capstone.  
For academic use only.
