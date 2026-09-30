"""Training and evaluation samples with consistent image and box transforms.

Images are read as OpenCV BGR ``uint8`` and converted explicitly to RGB
``float32`` in ``[0, 1]``, which is the input contract of torchvision
detectors (they normalise internally). Target boxes carry model labels
``1..K``; ignore regions carry ``IGNORE_LABEL`` so that the torchvision
proposal sampler never selects proposals matched to them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from sludge_micro.config import AugmentConfig, ProjectConfig
from sludge_micro.geometry import hflip_boxes, rot90_boxes, vflip_boxes
from sludge_micro.imaging import bgr_to_rgb_float, read_bgr
from sludge_micro.messages import ProjectError
from sludge_micro.prepare import IGNORE_LABEL
from sludge_micro.runtime import worker_seed

Target = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Sample:
    """One image with its boxes and labels in source pixel coordinates."""

    image_id: str
    path: Path
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]
    labels: tuple[int, ...]


def build_samples(config: ProjectConfig, images: pd.DataFrame, anns: pd.DataFrame) -> list[Sample]:
    """Create samples sorted by image id from prepared tables."""
    grouped = {k: v for k, v in anns.groupby("image_id", sort=True)}
    samples = []
    for row in images.sort_values("image_id", kind="stable").itertuples():
        part = grouped.get(row.image_id, anns.iloc[0:0])
        samples.append(
            Sample(
                image_id=row.image_id,
                path=config.resolve(row.path),
                width=int(row.width),
                height=int(row.height),
                boxes=tuple(map(tuple, part[["x1", "y1", "x2", "y2"]].to_numpy(float).tolist())),
                labels=tuple(int(v) for v in part["label"]),
            )
        )
    return samples


def augment_pair(
    image: np.ndarray, boxes: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the enabled flips and quarter turns to an image and its boxes together."""
    height, width = image.shape[:2]
    if cfg.hflip and rng.random() < cfg.flip_probability:
        image, boxes = image[:, ::-1], hflip_boxes(boxes, width)
    if cfg.vflip and rng.random() < cfg.flip_probability:
        image, boxes = image[::-1], vflip_boxes(boxes, height)
    if cfg.rotation90:
        k = int(rng.integers(0, 4))
        image, boxes = np.rot90(image, k), rot90_boxes(boxes, width, height, k)
    return np.ascontiguousarray(image), boxes


class DetectionDataset(Dataset):
    """Map-style dataset returning ``(image_tensor, target)`` pairs."""

    def __init__(self, samples: Sequence[Sample], augment: AugmentConfig | None, seed: int) -> None:
        """Store samples; augmentation is applied only when ``augment`` is given."""
        self.samples = tuple(samples)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the augmentation stream of an epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Target]:
        """Read, convert and optionally augment one sample."""
        sample = self.samples[index]
        image = read_bgr(sample.path)
        if image.shape[:2] != (sample.height, sample.width):
            raise ProjectError("input_contract", details=f"{sample.image_id} shape={image.shape}")
        rgb = bgr_to_rgb_float(image)
        boxes = np.asarray(sample.boxes, dtype=np.float64).reshape(-1, 4)
        if self.augment is not None:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            rgb, boxes = augment_pair(rgb, boxes, rng, self.augment)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(sample.labels, dtype=torch.int64),
        }
        return tensor, target


def collate(batch: list[tuple[torch.Tensor, Target]]) -> tuple[tuple, tuple]:
    """Keep images of different sizes as a tuple, as torchvision detectors expect."""
    images, targets = zip(*batch, strict=True)
    return images, targets


def class_presence(samples: Sequence[Sample]) -> Counter:
    """Number of the given images that contain each target label."""
    return Counter(label for s in samples for label in set(s.labels) if label > 0)


def sample_weights(samples: Sequence[Sample], policy: str) -> list[float] | None:
    """Weight each image by the inverse image frequency of its rarest target class.

    Frequencies come only from the given (training) samples. Images without
    target classes get the weight of the most frequent class.
    """
    if policy == "none":
        return None
    presence = class_presence(samples)
    floor = 1.0 / max(presence.values()) if presence else 1.0
    return [
        (
            1.0 / min(presence[label] for label in set(s.labels) if label > 0)
            if any(label > 0 for label in s.labels)
            else floor
        )
        for s in samples
    ]


def make_loader(
    samples: Sequence[Sample], config: ProjectConfig, train: bool, generator: torch.Generator
) -> DataLoader:
    """Build a deterministic loader; sampling and augmentation only for training."""
    dataset = DetectionDataset(
        samples, config.train.augment if train else None, config.runtime.seed
    )
    weights = sample_weights(samples, config.train.sampling) if train else None
    if weights is not None:
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=generator
        )
    elif train:
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size if train else 1,
        sampler=sampler,
        num_workers=config.runtime.loader_workers,
        collate_fn=collate,
        worker_init_fn=worker_seed,
        generator=generator,
    )


__all__ = [
    "IGNORE_LABEL",
    "DetectionDataset",
    "Sample",
    "build_samples",
    "class_presence",
    "make_loader",
]
