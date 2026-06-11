# =============================================================================
# 🌸  YOLOV8 ADDON — Track A Computer Vision Capstone Upgrade
# =============================================================================
# Adds YOLOv8 to your existing ResNet-18 project to create a proper
# TWO-STAGE COMPUTER VISION PIPELINE:
#
#   Stage 1 → YOLOv8  : Detects WHERE the flower is (bounding box)
#   Stage 2 → ResNet-18: Classifies WHAT type of flower it is
#
# Run from your Downloads folder AFTER flower_classifier.py has finished:
#   python yolo_addon.py
#
# Requirements:
#   python -m pip install ultralytics
# =============================================================================

import os
import time
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets

from ultralytics import YOLO

print("✅  Libraries loaded")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATASET_DIR  = "flower_photos"     # your existing dataset
MODEL_PATH   = "resnet18_best.pth" # your trained ResNet-18
YOLO_EPOCHS  = 10                  # H200 UPGRADE → 30
IMG_SIZE     = 224
SEED         = 42
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️   Device: {device}\n")

CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')
])
NUM_CLASSES = len(CLASS_NAMES)
EMOJI = {'daisy':'🌼','dandelion':'🌻','roses':'🌹',
         'sunflowers':'🌞','tulips':'🌷'}


# =============================================================================
# PART 1 — TRAIN YOLOV8 CLASSIFIER ON FLOWER DATASET
# =============================================================================
# YOLOv8 has a built-in classification mode (yolov8-cls) that accepts the
# exact same folder structure as your ResNet-18 dataset — no extra setup.
# =============================================================================

print("=" * 60)
print("PART 1 — Training YOLOv8 Classifier on Flower Dataset")
print("=" * 60)

# Split dataset into train/val for YOLO (it expects train/ and val/ folders)
YOLO_DATA = "yolo_flowers"

if not os.path.exists(YOLO_DATA):
    print("📂  Preparing YOLO dataset split ...")
    for split in ['train', 'val']:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(YOLO_DATA, split, cls), exist_ok=True)

    for cls in CLASS_NAMES:
        src = os.path.join(DATASET_DIR, cls)
        imgs = [f for f in os.listdir(src)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(imgs)
        split_idx = int(0.8 * len(imgs))
        for i, img in enumerate(imgs):
            split = 'train' if i < split_idx else 'val'
            shutil.copy(
                os.path.join(src, img),
                os.path.join(YOLO_DATA, split, cls, img)
            )
    print(f"✅  YOLO dataset ready in '{YOLO_DATA}/'")
else:
    print(f"✅  YOLO dataset already exists — skipping split")

# Train YOLOv8n-cls (nano = fastest, good for CPU demo)
print(f"\n🚀  Training YOLOv8n-cls for {YOLO_EPOCHS} epochs ...")
print("    (nano model — fast on CPU, ~10-20 min)\n")

yolo_clf = YOLO('yolov8n-cls.pt')   # downloads ~6 MB pretrained weights

t0 = time.time()
yolo_clf.train(
    data=YOLO_DATA,
    epochs=YOLO_EPOCHS,
    imgsz=IMG_SIZE,
    batch=16,           # H200 UPGRADE → 64
    workers=0,          # H200 UPGRADE → 4
    project='yolo_runs',
    name='flower_cls',
    exist_ok=True,
    verbose=False,
)
yolo_time = (time.time() - t0) / 60
print(f"\n✅  YOLOv8 training done in {yolo_time:.1f} min")

# Evaluate YOLOv8 on validation set
print("\n📊  Evaluating YOLOv8 on validation set ...")
yolo_val = yolo_clf.val(data=YOLO_DATA, verbose=False)
yolo_acc  = yolo_val.top1 * 100      # top-1 accuracy
print(f"    YOLOv8 val accuracy: {yolo_acc:.2f}%")


# =============================================================================
# PART 2 — LOAD RESNET-18 AND COMPARE
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 — Loading ResNet-18 for Comparison")
print("=" * 60)

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
resnet = resnet.to(device)
resnet.eval()
print(f"✅  ResNet-18 loaded from '{MODEL_PATH}'")

# Evaluate ResNet-18 on the same val split
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageFolder(
    os.path.join(YOLO_DATA, 'val'), transform=val_transform
)
val_loader  = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0
)

correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = resnet(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
resnet_acc = 100.0 * correct / total
print(f"    ResNet-18 val accuracy: {resnet_acc:.2f}%")


# =============================================================================
# PART 3 — COMPARISON BAR CHART
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 — Model Comparison Chart")
print("=" * 60)

BG   = '#1a1a2e'
CLR1 = '#4C72B0'
CLR2 = '#DD8452'

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

models_list = ['ResNet-18\n(Transfer Learning)', 'YOLOv8n-cls\n(Transfer Learning)']
accs        = [resnet_acc, yolo_acc]
colors      = [CLR1, CLR2]
bars        = ax.bar(models_list, accs, color=colors, width=0.4,
                     edgecolor='white', linewidth=1.5)

ax.set_ylabel('Validation Accuracy (%)', fontsize=12, color='white')
ax.set_title('Model Comparison — Flower Classification\nResNet-18 vs YOLOv8',
             fontsize=14, fontweight='bold', color='white', pad=15)
ax.set_ylim([0, 110])
ax.tick_params(colors='white', labelsize=11)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#444466')
ax.grid(axis='y', alpha=0.2, color='white')

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{acc:.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=13, color='white')

plt.tight_layout()
plt.savefig('yolo_vs_resnet.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("💾  Saved: yolo_vs_resnet.png")


# =============================================================================
# PART 4 — TWO-STAGE PIPELINE DEMO
# =============================================================================
# Stage 1: Pretrained YOLOv8 draws bounding boxes (detects objects)
# Stage 2: Our ResNet-18 classifies the flower type inside the box
#
# Even though YOLOv8 (COCO) wasn't trained on flowers specifically,
# it detects them as nearby categories (plant, vase, etc.)
# Our ResNet-18 then gives the precise flower classification.
# =============================================================================

print("\n" + "=" * 60)
print("PART 4 — Two-Stage Detection + Classification Pipeline Demo")
print("=" * 60)

# Load pretrained YOLOv8 detection model (COCO — 80 classes)
yolo_det = YOLO('yolov8n.pt')   # downloads ~6 MB

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

unnorm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


def pipeline_predict(image_path):
    """
    Two-stage pipeline:
      Stage 1 → YOLOv8 detects objects + draws bounding box
      Stage 2 → ResNet-18 classifies flower type in detected region
    """
    orig_img = Image.open(image_path).convert('RGB')
    W, H     = orig_img.size

    # ── Stage 1: YOLO Detection ───────────────────────────────
    det_results = yolo_det(image_path, verbose=False)[0]
    boxes = det_results.boxes

    # Use detected box if found, else full image
    if boxes is not None and len(boxes) > 0:
        # Pick the largest box (most likely the main flower)
        areas = [(b[2]-b[0]) * (b[3]-b[1])
                 for b in boxes.xyxy.cpu().numpy()]
        best  = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[best].astype(int)
        conf  = boxes.conf.cpu().numpy()[best]
        label = det_results.names[int(boxes.cls.cpu().numpy()[best])]
        detected = True
    else:
        # No detection → use whole image
        x1, y1, x2, y2 = 0, 0, W, H
        conf, label = 1.0, 'full image'
        detected = False

    # ── Stage 2: ResNet-18 Classification ────────────────────
    crop = orig_img.crop((x1, y1, x2, y2))
    tensor = infer_transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        out   = resnet(tensor)
        probs = torch.softmax(out, 1)[0].cpu()
        pred  = probs.argmax().item()

    flower_name = CLASS_NAMES[pred]
    flower_conf = probs[pred].item() * 100
    flower_emoji = EMOJI.get(flower_name, '🌺')

    return {
        'orig_img'    : orig_img,
        'box'         : (x1, y1, x2, y2),
        'yolo_label'  : label,
        'yolo_conf'   : conf,
        'detected'    : detected,
        'flower'      : flower_name,
        'flower_conf' : flower_conf,
        'emoji'       : flower_emoji,
        'probs'       : probs.numpy(),
    }


# Pick 6 random sample images (one per class + 1 extra)
sample_images = []
for cls in CLASS_NAMES:
    folder = os.path.join(DATASET_DIR, cls)
    imgs   = [f for f in os.listdir(folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    pick   = random.choice(imgs)
    sample_images.append(os.path.join(folder, pick))

random.shuffle(sample_images)
sample_images = sample_images[:6]

print(f"🔍  Running two-stage pipeline on {len(sample_images)} images ...")

results = [pipeline_predict(p) for p in sample_images]

# ── Visualization ──────────────────────────────────────────────
BG    = '#1a1a2e'
CARD  = '#16213e'
GREEN = '#00e676'
BLUE  = '#42a5f5'

fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=BG)
fig.suptitle(
    '🌸  Two-Stage CV Pipeline  —  YOLOv8 Detection  +  ResNet-18 Classification',
    fontsize=16, fontweight='bold', color='white', y=0.98
)

for ax, r in zip(axes.flat, results):
    ax.set_facecolor(CARD)
    ax.imshow(r['orig_img'])

    # Draw bounding box
    x1, y1, x2, y2 = r['box']
    W, H = r['orig_img'].size
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=3, edgecolor=GREEN, facecolor='none'
    )
    ax.add_patch(rect)

    # Stage 1 label (top of box)
    stage1_txt = (f"YOLO: {r['yolo_label']} ({r['yolo_conf']*100:.0f}%)"
                  if r['detected'] else "YOLO: full image")
    ax.text(x1 + 4, y1 - 8, stage1_txt,
            color='white', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=GREEN,
                      alpha=0.85, edgecolor='none'))

    # Stage 2 label (bottom of box)
    stage2_txt = f"ResNet-18: {r['flower']} ({r['flower_conf']:.0f}%)"
    ax.text(x1 + 4, y2 + 18, stage2_txt,
            color='white', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1565C0',
                      alpha=0.9, edgecolor='none'))

    ax.set_title(
        f"{r['emoji']}  {r['flower'].capitalize()}  —  {r['flower_conf']:.0f}% confident",
        color=GREEN, fontsize=11, fontweight='bold', pad=8
    )
    ax.axis('off')

    # Corner stage labels
    ax.text(0.01, 0.99, '① DETECT', transform=ax.transAxes,
            fontsize=7, color=GREEN, va='top', alpha=0.8)
    ax.text(0.01, 0.92, '② CLASSIFY', transform=ax.transAxes,
            fontsize=7, color=BLUE, va='top', alpha=0.8)

# Pipeline legend at bottom
from matplotlib.patches import FancyBboxPatch
fig.text(0.5, 0.01,
         '①  YOLOv8 (COCO pretrained)  →  detects object region  →  '
         '②  ResNet-18 (flower trained)  →  classifies flower type',
         ha='center', fontsize=11, color='#aaaacc', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('two_stage_pipeline.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("💾  Saved: two_stage_pipeline.png")


# =============================================================================
# PART 5 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("🏆  CAPSTONE PROJECT SUMMARY")
print("=" * 60)
print(f"""
  Two-Stage Computer Vision Pipeline
  ────────────────────────────────────────────────
  Stage 1  YOLOv8n (Detection)
           → Locates flower in image
           → Draws bounding box around region

  Stage 2  ResNet-18 (Classification)
           → Classifies flower type from cropped region
           → 5 classes: daisy, dandelion, roses,
             sunflowers, tulips

  ────────────────────────────────────────────────
  Model             Val Accuracy
  ResNet-18         {resnet_acc:.1f}%
  YOLOv8n-cls       {yolo_acc:.1f}%
  ────────────────────────────────────────────────
""")

print("📁  Files saved this run:")
for f in ['yolo_vs_resnet.png', 'two_stage_pipeline.png']:
    exists = '✅' if os.path.exists(f) else '⬜'
    print(f"   {exists}  {f}")

print("""
─────────────────────────────────────────────────
  How to describe this in your presentation:

  "I built a two-stage Computer Vision pipeline.
   Stage 1 uses YOLOv8 for real-time object detection,
   drawing bounding boxes around flowers in any image.
   Stage 2 feeds the detected region into a ResNet-18
   Transfer Learning classifier, identifying the exact
   flower species with {:.0f}% accuracy.
   Both models run on the NVIDIA H200 GPU."
─────────────────────────────────────────────────
""".format(resnet_acc))
