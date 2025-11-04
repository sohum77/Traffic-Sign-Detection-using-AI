# test_images.py
from ultralytics import YOLO
import os
import shutil
import sys

# ---------- CONFIG ----------
MODEL_PATH = r"runs/detect/train4/weights/best.pt"  # <- keep your best.pt path
TEST_DIR = r"data/test"                              # folder with your test images
VAL_IMAGES_DIR = r"data/val/images"                  # common validation images folder
OUT_PROJECT = r"runs/detect"
OUT_NAME = "test_results_clean"
CONF = 0.35     # lower -> more boxes, higher -> stricter
IOU = 0.45
IMGSZ = 640
MAX_DET = 100
COPY_FROM_VAL = True   # if test folder is empty and val images exist, copy some over

# ---------- helper: choose device ----------
def get_device():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Ultralytics accept "0" for GPU index or "cuda:0". We'll return "0".
            return "0"
    except Exception:
        pass
    return "cpu"

device = get_device()
print(f"Using device: {device}")

# ---------- ensure test dir exists ----------
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR, exist_ok=True)
    print(f"Created test folder: {TEST_DIR}")

# ---------- optionally copy images from validation to test ----------
def copy_from_val_if_needed(test_dir, val_dir, n_copy=8):
    try:
        if not os.path.exists(val_dir):
            return 0
        existing = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if len(existing) > 0:
            return 0
        imgs = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if not imgs:
            return 0
        n = min(n_copy, len(imgs))
        for fname in imgs[:n]:
            src = os.path.join(val_dir, fname)
            dst = os.path.join(test_dir, fname)
            shutil.copyfile(src, dst)
        return n
    except Exception as e:
        print("Warning while copying validation images:", e)
        return 0

if COPY_FROM_VAL:
    copied = copy_from_val_if_needed(TEST_DIR, VAL_IMAGES_DIR, n_copy=8)
    if copied > 0:
        print(f"Copied {copied} images from {VAL_IMAGES_DIR} -> {TEST_DIR}")

# ---------- list test images ----------
test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
print(f"Found {len(test_images)} test images in {TEST_DIR}")
if len(test_images) > 0:
    for f in test_images[:50]:
        print(" -", f)
else:
    print("No test images found. Put .jpg/.png files into", TEST_DIR)
    sys.exit(1)

# ---------- load model ----------
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Failed to load model. Check MODEL_PATH and that the file exists.")
    print("MODEL_PATH:", MODEL_PATH)
    print("Exception:", e)
    sys.exit(1)

print("Model classes:", model.names)

# ---------- run prediction ----------
try:
    results = model.predict(
        source=TEST_DIR,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        save=True,               # save annotated images
        project=OUT_PROJECT,
        name=OUT_NAME,
        show=False,
        device=device,           # auto-chosen device
        max_det=MAX_DET
    )
except Exception as e:
    print("Error during prediction:", e)
    sys.exit(1)

# ---------- print summary ----------
print("\n--- Detection Summary ---")
any_detections = False
for r in results:
    fname = os.path.basename(r.path) if hasattr(r, "path") else "unknown"
    nboxes = len(r.boxes) if hasattr(r, "boxes") else 0
    if nboxes == 0:
        print(f"{fname}: No detections")
    else:
        any_detections = True
        print(f"{fname}: {nboxes} boxes")
        # r.boxes.data: x1,y1,x2,y2,conf,class
        try:
            for b in r.boxes.data.tolist():
                x1,y1,x2,y2,conf,cls_id = b
                cls = model.names[int(cls_id)]
                print(f"  - {cls:20s} conf={conf:.2f} bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
        except Exception:
            print("  (could not print box details)")

saved_path = os.path.join(OUT_PROJECT, OUT_NAME)
print(f"\nSaved annotated images to: {saved_path}")

if not any_detections:
    print("\nNo detections were found on these images.")
    print("Possible reasons:")
    print(" - The images do not contain the trained sign classes.")
    print(" - The signs look very different from training images (angle, color, lighting).")
    print(" - Confidence threshold (CONF) is too high; try lowering it to 0.2.")
    print("\nSuggestions:")
    print(" - Test with images from data/val/images (these are from your dataset).")
    print(" - Lower CONF to 0.2 and re-run, or try smaller imgsz (e.g., 416).")
    print(" - Ensure your model (best.pt) is the one you trained and not an older/other file.")
else:
    print("\nDetections found — open the images in the saved folder to inspect the boxes.")
