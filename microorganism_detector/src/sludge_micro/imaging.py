"""Image decoding, colour contract and fingerprints."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from sludge_micro.messages import ProjectError
from sludge_micro.types import ImageFacts

EXIF_IFD = 0x8769
EXIF_MODEL = 0x0110
EXIF_DATETIME_ORIGINAL = 0x9003


def decode_bgr(data: bytes, name: str) -> np.ndarray:
    """Decode image bytes to an OpenCV BGR ``uint8`` array.

    Raises:
        ProjectError: If the bytes are not a decodable image.
    """
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ProjectError("image_decode_failed", path=name)
    return image


def read_bgr(path: Path) -> np.ndarray:
    """Read an image from a Unicode-safe path as BGR ``uint8``."""
    return decode_bgr(path.read_bytes(), str(path))


def check_bgr_contract(image: np.ndarray) -> None:
    """Verify the OpenCV input contract: ``HxWx3`` ``uint8`` array.

    Raises:
        ProjectError: If the array violates the contract.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ProjectError("input_contract", details=f"dtype={getattr(image, 'dtype', None)}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ProjectError("input_contract", details=f"shape={image.shape}")


def bgr_to_rgb_float(image: np.ndarray) -> np.ndarray:
    """Return a new RGB ``float32`` array in ``[0, 1]`` without touching the input."""
    check_bgr_contract(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def dhash(gray: np.ndarray, size: int) -> str:
    """Return a hexadecimal difference hash of a grayscale image."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.packbits(bits).tobytes().hex()


def hamming(a: str, b: str) -> int:
    """Return the Hamming distance of two hexadecimal hashes of equal length."""
    return int.bit_count(int(a, 16) ^ int(b, 16))


def exif_fields(data: bytes) -> tuple[str | None, str | None]:
    """Return ``DateTimeOriginal`` and camera model from EXIF, if present."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            exif = image.getexif()
    except (UnidentifiedImageError, OSError):
        return None, None
    moment = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL)
    model = exif.get(EXIF_MODEL)
    return (str(moment) if moment else None), (str(model) if model else None)


def image_facts(data: bytes, name: str, hash_size: int) -> ImageFacts:
    """Decode an image and compute its size, hashes and EXIF time.

    Args:
        data: Encoded image bytes.
        name: Name used in error messages.
        hash_size: Side of the difference hash grid.

    Returns:
        Immutable facts about the image.
    """
    image = decode_bgr(data, name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moment, model = exif_fields(data)
    return ImageFacts(
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        channels=int(image.shape[2]),
        file_sha256=hashlib.sha256(data).hexdigest(),
        pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        dhash=dhash(gray, hash_size),
        exif_datetime_original=moment,
        camera_model=model,
    )
