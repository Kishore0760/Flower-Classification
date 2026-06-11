# =============================================================================
# 🌸  LIVE FLOWER DETECTOR — Real-Time Webcam
# =============================================================================
# Uses your webcam to detect and classify flowers in real-time
# Stage 1 → YOLOv8  : draws bounding box around the flower
# Stage 2 → ResNet-18: classifies what type of flower it is
#
# CONTROLS:
#   Q          → Quit
#   S          → Save screenshot
#   SPACE      → Freeze/unfreeze frame
#
# RUN FROM YOUR DOWNLOADS FOLDER:
#   cd Downloads
#   python live_detector.py
# =============================================================================

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import time
import os

# ── Config ────────────────────────────────────────────────────
MODEL_PATH   = "resnet18_best.pth"
CLASS_NAMES  = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
NUM_CLASSES  = len(CLASS_NAMES)
EMOJI        = ["🌼", "🌻", "🌹", "🌞", "🌷"]
CAMERA_INDEX = 0        # Change to 1 if laptop camera doesn't open

# Colours (BGR format for OpenCV)
CLR_GREEN  = (0, 230, 118)
CLR_BLUE   = (255, 165, 0)
CLR_RED    = (80, 80, 255)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0, 0, 0)
CLR_DARK   = (26, 26, 46)
CLR_YELLOW = (0, 220, 220)

# Class colours for confidence bars
CLASS_COLORS = [
    (100, 220, 100),   # daisy      - green
    (100, 180, 255),   # dandelion  - orange
    (100, 100, 255),   # roses      - red
    (50,  200, 255),   # sunflowers - yellow
    (220, 100, 180),   # tulips     - pink
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️   Device: {device}")

# ── Load ResNet-18 ─────────────────────────────────────────────
print("📦  Loading ResNet-18 ...")
if not os.path.exists(MODEL_PATH):
    print(f"❌  Model file '{MODEL_PATH}' not found!")
    print("    Make sure you run this from the same Downloads folder")
    print("    where flower_classifier.py saved resnet18_best.pth")
    exit(1)

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
resnet = resnet.to(device)
resnet.eval()
print("✅  ResNet-18 loaded")

# ── Load YOLOv8 for detection ──────────────────────────────────
print("📦  Loading YOLOv8 ...")
yolo = YOLO('yolov8n.pt')
print("✅  YOLOv8 loaded")

# ── Image transform ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Open Webcam ────────────────────────────────────────────────
print(f"\n📷  Opening camera {CAMERA_INDEX} ...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"❌  Could not open camera {CAMERA_INDEX}")
    print("    Try changing CAMERA_INDEX = 1 at the top of this script")
    exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅  Camera opened: {W}x{H}")
print("\n🌸  Live detector running!")
print("    Q = Quit  |  S = Save screenshot  |  SPACE = Freeze frame\n")


# ── Helper: draw rounded rectangle ────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, r=8):
    cv2.line(img,  (x1+r, y1),   (x2-r, y1),   color, thickness)
    cv2.line(img,  (x1+r, y2),   (x2-r, y2),   color, thickness)
    cv2.line(img,  (x1,   y1+r), (x1,   y2-r), color, thickness)
    cv2.line(img,  (x2,   y1+r), (x2,   y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)


# ── Helper: overlay with transparency ─────────────────────────
def overlay_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    sub  = img[y1:y2, x1:x2]
    rect = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1-alpha, 0, sub)
    img[y1:y2, x1:x2] = sub


# ── Helper: draw confidence panel ─────────────────────────────
def draw_confidence_panel(frame, probs, pred_idx, px, py):
    panel_w = 220
    panel_h = NUM_CLASSES * 36 + 50
    overlay_rect(frame, px, py, px+panel_w, py+panel_h,
                 (26, 26, 46), alpha=0.75)
    draw_rounded_rect(frame, px, py, px+panel_w, py+panel_h,
                      CLR_BLUE, thickness=1)

    cv2.putText(frame, "CONFIDENCE", (px+8, py+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_BLUE, 1, cv2.LINE_AA)

    for i, (name, prob, clr) in enumerate(
            zip(CLASS_NAMES, probs, CLASS_COLORS)):
        y = py + 40 + i * 34
        bar_w = int(prob * 180)

        # Bar background
        cv2.rectangle(frame, (px+8, y), (px+8+180, y+18),
                       (60,60,80), -1)
        # Bar fill
        bar_color = CLR_GREEN if i == pred_idx else clr
        cv2.rectangle(frame, (px+8, y), (px+8+bar_w, y+18),
                       bar_color, -1)
        # Label
        label_clr = CLR_WHITE if i == pred_idx else (180, 180, 180)
        cv2.putText(frame, f"{name[:8]:<8} {prob*100:5.1f}%",
                    (px+10, y+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    label_clr, 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────
prev_time    = time.time()
fps_display  = 0.0
frozen       = False
frozen_frame = None
screenshot_n = 0

# Prediction state — updated every N frames for smoothness
pred_idx     = 0
probs        = np.ones(NUM_CLASSES) / NUM_CLASSES
box          = None
yolo_label   = ""
PRED_EVERY   = 3      # Run inference every 3 frames → smooth & fast
frame_count  = 0

while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Camera read failed — exiting")
            break
    else:
        frame = frozen_frame.copy()

    display = frame.copy()
    frame_count += 1

    # ── Run inference every PRED_EVERY frames ────────────────
    if frame_count % PRED_EVERY == 0 and not frozen:

        # Stage 1: YOLO detection
        det = yolo(frame, verbose=False)[0]
        boxes = det.boxes
        box = None

        if boxes is not None and len(boxes) > 0:
            areas = [(b[2]-b[0])*(b[3]-b[1])
                     for b in boxes.xyxy.cpu().numpy()]
            best  = int(np.argmax(areas))
            x1,y1,x2,y2 = boxes.xyxy.cpu().numpy()[best].astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W,x2), min(H,y2)
            if (x2-x1) > 20 and (y2-y1) > 20:
                box        = (x1, y1, x2, y2)
                yolo_label = det.names[int(boxes.cls.cpu().numpy()[best])]
                crop = frame[y1:y2, x1:x2]
            else:
                crop = frame
        else:
            crop = frame

        # Stage 2: ResNet-18 classification
        pil   = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inp   = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out    = resnet(inp)
            probs  = torch.softmax(out, 1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())

    # ── Draw bounding box ─────────────────────────────────────
    if box:
        x1, y1, x2, y2 = box
        conf = probs[pred_idx]
        if conf > 0.80:
            box_color = CLR_GREEN
        elif conf > 0.55:
            box_color = CLR_YELLOW
        else:
            box_color = CLR_RED

        # Glow effect — draw thick dim then thin bright
        cv2.rectangle(display, (x1-1,y1-1), (x2+1,y2+1),
                       tuple(c//3 for c in box_color), 4)
        cv2.rectangle(display, (x1,y1), (x2,y2), box_color, 2)

        # Stage labels on box
        cv2.putText(display, f"YOLO: {yolo_label}",
                    (x1+6, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

    # ── Main prediction label ─────────────────────────────────
    flower_name = CLASS_NAMES[pred_idx].upper()
    flower_emoji = EMOJI[pred_idx]
    conf_pct    = probs[pred_idx] * 100

    # Big label bar at top
    overlay_rect(display, 0, 0, W, 70, CLR_DARK, alpha=0.7)
    cv2.putText(display,
                f"  {flower_name}",
                (10, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, CLR_GREEN, 2, cv2.LINE_AA)
    cv2.putText(display,
                f"ResNet-18: {conf_pct:.1f}% confident",
                (420, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_WHITE, 1, cv2.LINE_AA)

    # ── Confidence panel (right side) ─────────────────────────
    draw_confidence_panel(display, probs, pred_idx,
                          W - 235, 80)

    # ── FPS counter ───────────────────────────────────────────
    now       = time.time()
    fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now-prev_time, 1e-6))
    prev_time = now
    cv2.putText(display, f"FPS: {fps_display:.1f}",
                (W-235, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)

    # ── Stage labels bottom ───────────────────────────────────
    overlay_rect(display, 0, H-38, W, H, CLR_DARK, alpha=0.6)
    cv2.putText(display,
                "  Stage 1: YOLOv8 (bounding box)    "
                "Stage 2: ResNet-18 (classification)    "
                "[Q] Quit   [S] Save   [SPACE] Freeze",
                (8, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,180), 1)

    # ── FROZEN indicator ─────────────────────────────────────
    if frozen:
        cv2.putText(display, "FROZEN — press SPACE to resume",
                    (W//2 - 200, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_YELLOW, 2, cv2.LINE_AA)

    # ── Show frame ───────────────────────────────────────────
    cv2.imshow("🌸 Live Flower Detector — YOLOv8 + ResNet-18", display)

    # ── Key handling ─────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:      # Q or ESC → quit
        print("\n👋  Closing detector ...")
        break

    elif key == ord('s'):                  # S → save screenshot
        screenshot_n += 1
        fname = f"live_capture_{screenshot_n:02d}_{CLASS_NAMES[pred_idx]}.png"
        cv2.imwrite(fname, display)
        print(f"📸  Screenshot saved: {fname}")

    elif key == ord(' '):                  # SPACE → freeze/unfreeze
        frozen = not frozen
        if frozen:
            frozen_frame = frame.copy()
            print(f"⏸   Frame frozen — prediction: "
                  f"{CLASS_NAMES[pred_idx]} ({conf_pct:.1f}%)")
        else:
            print("▶️   Resumed")

cap.release()
cv2.destroyAllWindows()
print("\n✅  Detector closed")
print(f"   Screenshots saved: {screenshot_n}")
