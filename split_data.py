import os
import random
import shutil

# Paths (keep same as your project)
image_dir = "dataset/images"
label_dir = "dataset/labels"

train_img_dir = "data/train/images"
train_lbl_dir = "data/train/labels"
val_img_dir = "data/val/images"
val_lbl_dir = "data/val/labels"

# Make folders if not exist
for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Get all image files (jpg/png)
images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(images)

# Split 80% for training, 20% for validation
split_index = int(0.8 * len(images))
train_files = images[:split_index]
val_files = images[split_index:]

def copy_files(file_list, target_img_dir, target_lbl_dir):
    for img in file_list:
        lbl = os.path.splitext(img)[0] + ".txt"
        if os.path.exists(os.path.join(label_dir, lbl)):  # only if label exists
            shutil.copy(os.path.join(image_dir, img), os.path.join(target_img_dir, img))
            shutil.copy(os.path.join(label_dir, lbl), os.path.join(target_lbl_dir, lbl))

copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(val_files, val_img_dir, val_lbl_dir)

print("✅ Dataset split complete! Files moved into data/train and data/val.")
print(f"Training images: {len(train_files)} | Validation images: {len(val_files)}")
