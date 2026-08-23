"""Perceptual hashing adapter using imagehash."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import imagehash
    from PIL import Image

    _IMAGEHASH_AVAILABLE = True
except Exception:  # pragma: no cover - imagehash may be missing in minimal envs
    imagehash = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment, misc]
    _IMAGEHASH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerceptualHasher:
    """Compute pHash values for images."""

    def __init__(self) -> None:
        if not _IMAGEHASH_AVAILABLE:
            raise RuntimeError(
                "imagehash and Pillow are required for perceptual hashing"
            )

    def compute(self, path: Path) -> str | None:
        """Return the pHash of *path* as a hex string, or None on failure."""
        try:
            with Image.open(path) as img:  # type: ignore[attr-defined]
                # Convert to RGB to normalise across formats (e.g. HEIC, PNG).
                rgb = img.convert("RGB")
                return str(imagehash.phash(rgb))
        except Exception as exc:
            logger.warning("Could not compute perceptual hash for %s: %s", path, exc)
            return None

    def distance(self, hash_a: str, hash_b: str) -> int:
        """Return the Hamming distance between two pHash hex strings."""
        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)  # type: ignore[attr-defined]
