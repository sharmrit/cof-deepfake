"""
Dataset loaders for deepfake detection benchmarks.

Supported datasets:
  - FaceForensics++ (FF++): c23 compression, 4 manipulation methods
  - CelebDF (v2): Celebrity face swaps
  - DFDC: Facebook Deepfake Detection Challenge
"""

import os
import glob
import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_eval_transforms


class BaseFaceDataset(Dataset):
    """Base class for face forensic datasets."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.transform = transform

        if max_samples_per_class is not None:
            real_paths = [p for p, l in zip(image_paths, labels) if l == 0]
            fake_paths = [p for p, l in zip(image_paths, labels) if l == 1]
            random.shuffle(real_paths)
            random.shuffle(fake_paths)
            real_paths = real_paths[:max_samples_per_class]
            fake_paths = fake_paths[:max_samples_per_class]
            self.image_paths = real_paths + fake_paths
            self.labels = [0] * len(real_paths) + [1] * len(fake_paths)
        else:
            self.image_paths = list(image_paths)
            self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a blank image on read failure
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FaceForensicsDataset(BaseFaceDataset):
    """
    FaceForensics++ dataset loader.

    Expected directory layout::

        root/
        ├── original_sequences/youtube/c23/frames/
        │   ├── 000/
        │   │   ├── 0000.png
        │   │   └── ...
        │   └── ...
        └── manipulated_sequences/
            ├── Deepfakes/c23/frames/
            ├── Face2Face/c23/frames/
            ├── FaceSwap/c23/frames/
            └── NeuralTextures/c23/frames/
    """

    MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        compression: str = "c23",
        manipulations: Optional[List[str]] = None,
        transform=None,
        max_samples_per_class: Optional[int] = None,
        split_file: Optional[str] = None,
    ):
        self.root = Path(root)
        self.compression = compression
        self.manipulations = manipulations or self.MANIPULATIONS

        # Load split video IDs
        video_ids = self._load_split(split, split_file)

        image_paths, labels = [], []

        # Real frames — batch scan instead of per-video glob
        real_dir = self.root / "original_sequences" / "youtube" / compression / "frames"
        video_set = set(video_ids)
        if real_dir.exists():
            for vid_dir in sorted(real_dir.iterdir()):
                if vid_dir.is_dir() and vid_dir.name in video_set:
                    frames = sorted([str(f) for f in vid_dir.iterdir() if f.suffix == ".png"])
                    image_paths.extend(frames)
                    labels.extend([0] * len(frames))

        # Fake frames
        for manip in self.manipulations:
            fake_dir = (
                self.root / "manipulated_sequences" / manip / compression / "frames"
            )
            if not fake_dir.exists():
                continue
            for vid_dir in sorted(fake_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue
                # FF++ paired naming: check if either part of name matches
                parts = vid_dir.name.split("_")
                if any(p in video_set for p in parts):
                    frames = sorted([str(f) for f in vid_dir.iterdir() if f.suffix == ".png"])
                    image_paths.extend(frames)
                    labels.extend([1] * len(frames))

        super().__init__(image_paths, labels, transform, max_samples_per_class)

    def _load_split(
        self, split: str, split_file: Optional[str] = None
    ) -> List[str]:
        """Load train/val/test video ID split."""
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                return [line.strip() for line in f if line.strip()]

        # Default FF++ split files
        split_dir = self.root / "splits"
        if split_dir.exists():
            split_path = split_dir / f"{split}.json"
            if split_path.exists():
                import json
                with open(split_path) as f:
                    pairs = json.load(f)
                ids = set()
                for pair in pairs:
                    ids.update(pair)
                return list(ids)

        # Fallback: use all available video directories
        real_dir = (
            self.root
            / "original_sequences"
            / "youtube"
            / self.compression
            / "frames"
        )
        if real_dir.exists():
            all_ids = sorted([d.name for d in real_dir.iterdir() if d.is_dir()])
            n = len(all_ids)
            if split == "train":
                return all_ids[: int(0.72 * n)]
            elif split == "val":
                return all_ids[int(0.72 * n) : int(0.86 * n)]
            else:
                return all_ids[int(0.86 * n) :]
        return []


class CelebDFDataset(BaseFaceDataset):
    """
    CelebDF (v2) dataset loader.

    Expected directory layout::

        root/
        ├── Celeb-real/
        │   ├── id0_0000.png
        │   └── ...
        ├── Celeb-synthesis/
        │   ├── id0_id1_0000.png
        │   └── ...
        └── List_of_testing_videos.txt
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform=None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        image_paths, labels = [], []

        real_dir = self.root / "Celeb-real"
        fake_dir = self.root / "Celeb-synthesis"

        if real_dir.exists():
            for ext in ["png", "jpg"]:
                frames = sorted([str(p) for p in real_dir.rglob(f"*.{ext}")])
                image_paths.extend(frames)
                labels.extend([0] * len(frames))

        if fake_dir.exists():
            for ext in ["png", "jpg"]:
                frames = sorted([str(p) for p in fake_dir.rglob(f"*.{ext}")])
                image_paths.extend(frames)
                labels.extend([1] * len(frames))

        super().__init__(image_paths, labels, transform, max_samples_per_class)


class DFDCDataset(BaseFaceDataset):
    """
    DFDC (Deepfake Detection Challenge) dataset loader.

    Searches for real/fake directories in common DFDC layouts::

        root/
        ├── dfdc_faces/train/real/     ← primary layout
        │   └── *.png
        ├── dfdc_faces/train/fake/
        │   └── *.png
        ├── frames/real/               ← alternative layout
        └── frames/fake/

    Uses rglob to find images in nested subdirectories.
    """

    # Candidate (real_subdir, fake_subdir) relative to root
    _LAYOUT_CANDIDATES = [
        ("dfdc_faces/train/real", "dfdc_faces/train/fake"),
        ("dfdc_faces/test/real",  "dfdc_faces/test/fake"),
        ("frames/real",           "frames/fake"),
        ("real",                  "fake"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform=None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        image_paths, labels = [], []

        # Auto-detect directory layout
        real_dir = fake_dir = None
        for real_sub, fake_sub in self._LAYOUT_CANDIDATES:
            r = self.root / real_sub
            f = self.root / fake_sub
            if r.exists() and f.exists():
                real_dir, fake_dir = r, f
                break

        if real_dir is None:
            # Last resort: search for any "real" / "fake" directories
            for d in self.root.rglob("real"):
                if d.is_dir():
                    real_dir = d
                    break
            for d in self.root.rglob("fake"):
                if d.is_dir():
                    fake_dir = d
                    break

        if real_dir and real_dir.exists():
            for ext in ["png", "jpg"]:
                frames = sorted([str(p) for p in real_dir.rglob(f"*.{ext}")])
                image_paths.extend(frames)
                labels.extend([0] * len(frames))

        if fake_dir and fake_dir.exists():
            for ext in ["png", "jpg"]:
                frames = sorted([str(p) for p in fake_dir.rglob(f"*.{ext}")])
                image_paths.extend(frames)
                labels.extend([1] * len(frames))

        super().__init__(image_paths, labels, transform, max_samples_per_class)


def get_dataset(
    name: str,
    root: str,
    split: str = "test",
    transform=None,
    max_samples_per_class: Optional[int] = None,
) -> BaseFaceDataset:
    """Get dataset by name."""
    registry = {
        "faceforensics": FaceForensicsDataset,
        "celebdf": CelebDFDataset,
        "dfdc": DFDCDataset,
    }
    if name not in registry:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(registry)}")
    return registry[name](
        root=root,
        split=split,
        transform=transform,
        max_samples_per_class=max_samples_per_class,
    )
