"""Consistent image and box transforms, ignore regions and deterministic loading."""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest
import torch

from sludge_micro.config import AugmentConfig
from sludge_micro.dataset import (
    DetectionDataset,
    Sample,
    augment_pair,
    make_loader,
    sample_weights,
)
from sludge_micro.model import build_detector_raw
from sludge_micro.prepare import IGNORE_LABEL


def marked_image(height=30, width=50):
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[5:12, 8:30] = 1.0
    return image, np.array([[8.0, 5.0, 30.0, 12.0]])


def pixel_box(image):
    ys, xs = np.nonzero(image[:, :, 0])
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_augmentation_moves_boxes_with_pixels(flags):
    cfg = AugmentConfig(rotation90=flags[0], hflip=flags[1], vflip=flags[2], flip_probability=0.5)
    for seed in range(8):
        image, boxes = marked_image()
        out, moved = augment_pair(image, boxes, np.random.default_rng(seed), cfg)
        assert moved[0] == pytest.approx(pixel_box(out))
        assert out.flags["C_CONTIGUOUS"]


def write_sample(tmp_path, helpers, name="s.bmp", labels=(3, IGNORE_LABEL)):
    image = helpers.synthetic_image(7, 64, 48)
    path = tmp_path / name
    cv2.imwrite(str(path), image)
    boxes = ((1.0, 2.0, 20.0, 30.0), (30.0, 5.0, 60.0, 40.0))[: len(labels)]
    return Sample("s", path, 64, 48, boxes, tuple(labels)), image


def test_dataset_returns_rgb_float_and_int64_labels(tmp_path, helpers):
    sample, image = write_sample(tmp_path, helpers)
    before = sample.path.read_bytes()
    tensor, target = DetectionDataset([sample], None, 0)[0]
    assert tensor.dtype == torch.float32 and tuple(tensor.shape) == (3, 48, 64)
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
    assert torch.allclose(tensor[0], torch.from_numpy(image[:, :, 2] / 255.0).float())
    assert target["labels"].dtype == torch.int64
    assert target["labels"].tolist() == [3, IGNORE_LABEL]
    assert sample.path.read_bytes() == before


def test_ignore_regions_are_never_sampled_by_the_box_head():
    model = build_detector_raw("fasterrcnn_resnet50_fpn_v2", 8, 64, 64)
    heads = model.roi_heads
    heads.fg_bg_sampler.batch_size_per_image = 512
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0], [40.0, 40.0, 60.0, 60.0]]),
        "labels": torch.tensor([3, IGNORE_LABEL]),
    }
    proposals = [
        torch.tensor([[1.0, 1.0, 20.0, 20.0], [41.0, 41.0, 60.0, 60.0], [0.0, 40.0, 10.0, 50.0]])
    ]
    sampled, _, labels, _ = heads.select_training_samples(proposals, [target])
    assert IGNORE_LABEL not in labels[0].tolist()
    assert sorted(labels[0].tolist()) == [0, 3, 3]
    kept = {tuple(round(v) for v in box) for box in sampled[0].tolist()}
    assert (41, 41, 60, 60) not in kept and (40, 40, 60, 60) not in kept


def sample(image_id, labels):
    return Sample(image_id, None, 1, 1, tuple((0.0, 0.0, 1.0, 1.0) for _ in labels), tuple(labels))


def test_rarest_class_weights_use_only_given_samples():
    samples = [sample("a", [1, 1]), sample("b", [1, 2]), sample("c", [1]), sample("d", [])]
    weights = sample_weights(samples, "rarest_class_presence")
    assert weights == pytest.approx([1 / 3, 1.0, 1 / 3, 1 / 3])
    assert sample_weights(samples, "none") is None


def test_loader_order_and_augmentation_repeat_for_one_seed(tmp_path, helpers, config):
    samples = [write_sample(tmp_path, helpers, f"{i}.bmp", (3,))[0] for i in range(4)]
    augment = AugmentConfig(rotation90=True, hflip=True, vflip=True, flip_probability=0.5)
    cfg = config.model_copy(update={"train": config.train.model_copy(update={"augment": augment})})

    def run():
        generator = torch.Generator().manual_seed(5)
        loader = make_loader(samples, cfg, True, generator)
        loader.dataset.set_epoch(1)
        return [(img[0].clone(), tgt[0]["boxes"].clone()) for img, tgt in loader]

    first, second = run(), run()
    for (a_img, a_box), (b_img, b_box) in zip(first, second, strict=True):
        assert torch.equal(a_img, b_img) and torch.equal(a_box, b_box)
