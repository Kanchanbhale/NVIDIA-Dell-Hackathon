import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
# Assume this script lives in: ROOT/scripts/build_dataset.py
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

TMP_IMAGES = OUT / "tmp_images"   # temporary flat image store
TMP_LABELS = OUT / "tmp_labels"   # temporary flat label store

FINAL_IMAGES = OUT / "images"     # final /images/train, /images/val
FINAL_LABELS = OUT / "labels"     # final /labels/train, /labels/val

TRAIN_RATIO = 0.8  # 80/20 split


# ---------------------------------------------------------
# Utility: reset output folders
# ---------------------------------------------------------
def reset_output_dirs():
    if OUT.exists():
        shutil.rmtree(OUT)
    TMP_IMAGES.mkdir(parents=True, exist_ok=True)
    TMP_LABELS.mkdir(parents=True, exist_ok=True)
    (FINAL_IMAGES / "train").mkdir(parents=True, exist_ok=True)
    (FINAL_IMAGES / "val").mkdir(parents=True, exist_ok=True)
    (FINAL_LABELS / "train").mkdir(parents=True, exist_ok=True)
    (FINAL_LABELS / "val").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Utility: YOLO label writer (full-image fall class)
# ---------------------------------------------------------
def write_full_image_label(label_path: Path, class_id: int = 0):
    """
    Writes a YOLO-format label where the whole image is one box:
    <class> <x_center> <y_center> <w> <h>
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


# ---------------------------------------------------------
# 1) Use existing YOLO dataset: fall_detection_images
# ---------------------------------------------------------
def process_fall_detection_images() -> list[tuple[Path, Path]]:
    """
    Collect all (image, label) pairs from data/raw/fall_detection_images
    which already has YOLO-style structure:

        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt

    Returns list of (image_path, label_path) that were copied into TMP_*.
    """
    src_root = RAW / "fall_detection_images"
    pairs: list[tuple[Path, Path]] = []

    if not src_root.exists():
        print("[1] fall_detection_images: folder not found, skipping.")
        return pairs

    print("[1] Using existing YOLO dataset: fall_detection_images")

    for split in ["train", "val"]:
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            label_name = img_path.stem + ".txt"
            lbl_path = lbl_dir / label_name
            if not lbl_path.exists():
                print(f"   [WARN] Missing label for {img_path}, skipping.")
                continue

            # Copy into temporary flat structure
            new_img = TMP_IMAGES / img_path.name
            new_lbl = TMP_LABELS / label_name
            shutil.copy2(img_path, new_img)
            shutil.copy2(lbl_path, new_lbl)
            pairs.append((new_img, new_lbl))

    print(f"[1] fall_detection_images: {len(pairs)} images collected.")
    return pairs


# ---------------------------------------------------------
# 2) LE2I video dataset (Coffee_room_01 / Home_01 / etc.)
# ---------------------------------------------------------
def read_le2i_annotation(txt_path: Path) -> tuple[int, int] | None:
    """
    Each txt file has at least 2 numbers (start_frame, end_frame)
    for the fall segment, as used in many LE2I scripts.

    We read those first two numbers.
    """
    try:
        data = pd.read_table(txt_path, header=None, delim_whitespace=True)
        start = int(data.iloc[0, 0])
        end = int(data.iloc[1, 0])
        return start, end
    except Exception as e:
        print(f"   [WARN] Failed to parse annotation {txt_path}: {e}")
        return None


def extract_le2i_frames() -> list[tuple[Path, Path]]:
    """
    For each scene folder under data/raw/le2i:

        Coffee_room_01/
            Annotation_files/video (1).txt
            Videos/video (1).avi
            ...

    We:
      - read annotation → (fall_start, fall_end)
      - grab 4 frames: start, start+10, end, end+10
      - save each frame as an image
      - create a YOLO label (full-frame fall) for each

    Returns list of (image_path, label_path) created in TMP_*.
    """
    le2i_root = RAW / "le2i"
    pairs: list[tuple[Path, Path]] = []

    if not le2i_root.exists():
        print("[2] LE2I: folder not found, skipping.")
        return pairs

    print("[2] Processing LE2I videos...")

    for scene_dir in sorted(le2i_root.iterdir()):
        if not scene_dir.is_dir():
            continue

        ann_dir = scene_dir / "Annotation_files"
        vid_dir = scene_dir / "Videos"
        if not (ann_dir.exists() and vid_dir.exists()):
            print(f"   [WARN] Missing Annotation_files or Videos in {scene_dir.name}, skipping.")
            continue

        print(f"   Scene: {scene_dir.name}")

        for video_path in sorted(vid_dir.glob("*.avi")):
            base = video_path.stem  # e.g. "video (1)"
            txt_path = ann_dir / f"{base}.txt"
            if not txt_path.exists():
                print(f"      [WARN] No annotation for {video_path.name}, skipping.")
                continue

            fall_range = read_le2i_annotation(txt_path)
            if fall_range is None:
                continue

            start_frame, end_frame = fall_range
            target_frames = {
                start_frame,
                start_frame + 10,
                end_frame,
                end_frame + 10,
            }

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"      [WARN] Could not open {video_path.name}")
                continue

            frame_idx = 0
            grabbed_for_this_video = 0

            while cap.isOpened() and target_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx in target_frames:
                    img_name = f"{scene_dir.name}_{base.replace(' ', '_')}_f{frame_idx:05d}.jpg"
                    img_path = TMP_IMAGES / img_name
                    cv2.imwrite(str(img_path), frame)

                    lbl_path = TMP_LABELS / (img_path.stem + ".txt")
                    write_full_image_label(lbl_path, class_id=0)

                    pairs.append((img_path, lbl_path))
                    grabbed_for_this_video += 1
                    target_frames.remove(frame_idx)

                frame_idx += 1

            cap.release()
            if grabbed_for_this_video == 0:
                print(f"      [WARN] No frames captured for {video_path.name}")

    print(f"[2] LE2I: {len(pairs)} frames extracted & labeled.")
    return pairs


# ---------------------------------------------------------
# 3) Combine & train/val split
# ---------------------------------------------------------
def split_train_val(pairs: list[tuple[Path, Path]]):
    if not pairs:
        print("❌ No samples collected. Check your raw data folders.")
        return

    image_paths = np.array([p[0] for p in pairs])
    label_paths = np.array([p[1] for p in pairs])

    train_idx, val_idx = train_test_split(
        np.arange(len(image_paths)),
        test_size=1.0 - TRAIN_RATIO,
        shuffle=True,
        random_state=42,
    )

    def copy_subset(indices, split_name: str):
        img_dest = FINAL_IMAGES / split_name
        lbl_dest = FINAL_LABELS / split_name
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        for i in indices:
            src_img = image_paths[i]
            src_lbl = label_paths[i]
            dst_img = img_dest / src_img.name
            dst_lbl = lbl_dest / src_lbl.name
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    copy_subset(train_idx, "train")
    copy_subset(val_idx, "val")

    print(f"[✓] Train images: {len(train_idx)}")
    print(f"[✓] Val images  : {len(val_idx)}")


# ---------------------------------------------------------
# 4) Main
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚧 Building unified YOLO dataset...\n")

    reset_output_dirs()

    pairs_all: list[tuple[Path, Path]] = []

    # 1) existing YOLO dataset
    pairs_all.extend(process_fall_detection_images())

    # 2) LE2I video dataset
    pairs_all.extend(extract_le2i_frames())

    print("\n---------------------------------------------")
    print(f"Total samples before split: {len(pairs_all)}")
    print("---------------------------------------------\n")

    split_train_val(pairs_all)

    print("\n✅ DONE!")
    print(f"   Images: {FINAL_IMAGES}")
    print(f"   Labels: {FINAL_LABELS}")
    print("\nNow you can train YOLO with fall.yaml.")
    
    # Optional: clean temporary folders
    shutil.rmtree(TMP_IMAGES, ignore_errors=True)
    shutil.rmtree(TMP_LABELS, ignore_errors=True)
