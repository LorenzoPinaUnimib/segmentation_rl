"""
dataset_inspector.py
Automatically inspects a dataset root directory and finds (image, mask) pairs.
Handles: flat dirs, nested dirs, CSV-based, COCO JSON annotations.
"""
import os
import re
import csv
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

# Image extensions we accept
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Keywords that suggest a file is a mask (not an image)
MASK_KEYWORDS = re.compile(
    r"(mask|label|seg|annotation|gt|ground_truth|truth)",
    re.IGNORECASE
)

# COCO annotation file names to search for
COCO_JSON_NAMES = {"_annotations.coco.json", "_annotations_coco.json",
                   "annotations.json", "_annotations.json"}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level file discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_all_images(root: str) -> List[Path]:
    """Recursively find all image files under root."""
    root = Path(root)
    imgs = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs


def is_mask_path(p: Path) -> bool:
    """Heuristic: does the path name look like a mask?"""
    return bool(MASK_KEYWORDS.search(str(p)))


# ─────────────────────────────────────────────────────────────────────────────
# COCO JSON discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_coco_jsons(root: Path) -> List[Path]:
    """Find all COCO annotation JSON files under root (any depth)."""
    found = []
    for p in sorted(root.rglob("*.json")):
        if p.name.lower() in COCO_JSON_NAMES or "annotation" in p.name.lower():
            # Quick sanity check: file contains 'images' and 'annotations' keys
            try:
                with open(p) as f:
                    data = json.load(f)
                if "images" in data and "annotations" in data:
                    found.append(p)
            except Exception:
                pass
    return found


def load_coco_pairs(json_path: Path) -> List[Tuple[Path, dict]]:
    """
    Parse a COCO JSON and return (image_path, annotation_dict) pairs.
    annotation_dict has keys: segmentation, bbox, category_id, image_h, image_w
    Images are expected to live in the same directory as the JSON.
    Returns only pairs where the image file actually exists on disk.
    """
    img_dir = json_path.parent
    with open(json_path) as f:
        data = json.load(f)

    # Build image_id → metadata map
    id_to_img: Dict[int, dict] = {img["id"]: img for img in data["images"]}

    # Build image_id → annotations list (one image can have multiple annotations)
    id_to_anns: Dict[int, List[dict]] = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        id_to_anns.setdefault(iid, []).append(ann)

    pairs = []
    for img_meta in data["images"]:
        iid = img_meta["id"]
        img_file = img_dir / img_meta["file_name"]
        if not img_file.exists():
            continue
        anns = id_to_anns.get(iid, [])
        if not anns:
            continue  # no annotation for this image → skip
        ann_info = {
            "annotations": anns,
            "height": img_meta["height"],
            "width":  img_meta["width"],
            "json_path": str(json_path),
        }
        pairs.append((img_file, ann_info))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Pair builders (image-file-based strategies, kept for non-COCO datasets)
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs_by_name(all_files: List[Path]) -> List[Tuple[Path, Path]]:
    masks = [f for f in all_files if is_mask_path(f)]
    images = [f for f in all_files if not is_mask_path(f)]

    def normalise(p: Path) -> str:
        name = p.stem
        return MASK_KEYWORDS.sub("", name).strip("_- ")

    mask_lookup = {}
    for m in masks:
        key = normalise(m)
        mask_lookup[key] = m

    pairs = []
    for img in images:
        key = normalise(img)
        if key in mask_lookup:
            pairs.append((img, mask_lookup[key]))
    return pairs


def build_pairs_by_parallel_dirs(root: Path) -> List[Tuple[Path, Path]]:
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    img_dirs  = [d for d in subdirs if not is_mask_path(d)]
    mask_dirs = [d for d in subdirs if is_mask_path(d)]

    if not img_dirs or not mask_dirs:
        return []

    pairs = []
    for img_dir in img_dirs:
        for mask_dir in mask_dirs:
            img_files  = {f.stem: f for f in img_dir.iterdir()
                          if f.suffix.lower() in IMG_EXTS}
            mask_files = {f.stem: f for f in mask_dir.iterdir()
                          if f.suffix.lower() in IMG_EXTS}
            common = set(img_files) & set(mask_files)
            for stem in sorted(common):
                pairs.append((img_files[stem], mask_files[stem]))
    return pairs


def build_pairs_split_dirs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for split in ["train", "val", "validation", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        p = build_pairs_by_parallel_dirs(split_dir)
        pairs.extend(p)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main inspector
# ─────────────────────────────────────────────────────────────────────────────

class DatasetInspector:
    """
    Given a root path, find all (image, annotation) pairs.
    Supports COCO JSON annotations and image-mask file pairs.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        assert self.root.exists(), f"Dataset root not found: {self.root}"
        self.coco_mode = False   # set True if COCO JSONs are found

    def inspect(self) -> List[Tuple[Path, object]]:
        """
        Returns a list of (image_path, annotation) tuples.
        If COCO mode: annotation is a dict with 'annotations', 'height', 'width'.
        If file-pair mode: annotation is a Path to the mask image file.
        """
        print(f"[inspector] Scanning: {self.root}")

        # ── Strategy 0: COCO JSON (highest priority) ─────────────────────────
        coco_jsons = find_coco_jsons(self.root)
        if coco_jsons:
            all_pairs = []
            for jpath in coco_jsons:
                pairs = load_coco_pairs(jpath)
                all_pairs.extend(pairs)
                print(f"[inspector] COCO JSON '{jpath.name}': {len(pairs)} annotated images.")
            if all_pairs:
                self.coco_mode = True
                print(f"[inspector] COCO mode — total {len(all_pairs)} pairs.")
                return all_pairs

        # ── Strategy 1: split dirs (train/val/test → images/masks) ──────────
        pairs = build_pairs_split_dirs(self.root)
        if pairs:
            print(f"[inspector] Strategy 'split dirs' found {len(pairs)} pairs.")
            return pairs

        # ── Strategy 2: parallel sibling dirs in root ────────────────────────
        pairs = build_pairs_by_parallel_dirs(self.root)
        if pairs:
            print(f"[inspector] Strategy 'parallel dirs' found {len(pairs)} pairs.")
            return pairs

        # ── Strategy 3: name-matching across all files ───────────────────────
        all_files = find_all_images(self.root)
        print(f"[inspector] Total image files found: {len(all_files)}")
        pairs = build_pairs_by_name(all_files)
        if pairs:
            print(f"[inspector] Strategy 'name match' found {len(pairs)} pairs.")
            return pairs

        print("[inspector] WARNING: Could not find mask pairs. "
              "Will use synthetic masks as fallback.")
        return []

    def save_pairs_csv(self, pairs: List[Tuple[Path, object]], out_path: str) -> None:
        """Save pairs to CSV. Works for both COCO and file-pair modes."""
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "mask_path_or_json"])
            for img, ann in pairs:
                if isinstance(ann, dict):
                    writer.writerow([str(img), ann.get("json_path", "")])
                else:
                    writer.writerow([str(img), str(ann)])
        print(f"[inspector] Pairs saved → {out_path}")

    @staticmethod
    def load_pairs_csv(csv_path: str) -> List[Tuple[Path, Path]]:
        pairs = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((Path(row["image_path"]),
                               Path(row["mask_path_or_json"])))
        return pairs