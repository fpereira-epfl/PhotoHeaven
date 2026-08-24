"""Perceptual hashing adapter for videos via sampled keyframes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

try:
    import cv2
    import imagehash
    from PIL import Image

    _VIDEO_HASH_AVAILABLE = True
except Exception:  # pragma: no cover - optional deps may be missing
    cv2 = None  # type: ignore[assignment]
    imagehash = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment, misc]
    _VIDEO_HASH_AVAILABLE = False

logger = logging.getLogger(__name__)


class VideoFrameHasher:
    """Compute pHashes for a small set of keyframes extracted from a video."""

    def __init__(self, frame_count: int = 3) -> None:
        if not _VIDEO_HASH_AVAILABLE:
            raise RuntimeError(
                "opencv-python, imagehash and Pillow are required for video hashing"
            )
        self.frame_count = frame_count

    def compute(self, path: Path) -> list[str] | None:
        """Return a list of pHash hex strings for sampled keyframes, or None."""
        try:
            cap = cv2.VideoCapture(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Could not open video %s: %s", path, exc)
            return None

        try:
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore[attr-defined]
            fps = cap.get(cv2.CAP_PROP_FPS)  # type: ignore[attr-defined]
            duration_seconds = frame_count_total / fps if fps else 0
            if frame_count_total <= 0 or duration_seconds <= 0:
                logger.warning("Could not determine video duration for %s", path)
                return None

            positions = self._sample_positions(duration_seconds)
            hashes: list[str] = []
            for position in positions:
                cap.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0)  # type: ignore[attr-defined]
                success, frame = cap.read()
                if not success:
                    continue
                try:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # type: ignore[attr-defined]
                    hashes.append(str(imagehash.phash(image)))
                except Exception as exc:
                    logger.debug(
                        "Could not hash keyframe at %.1fs for %s: %s",
                        position,
                        path,
                        exc,
                    )
            return hashes if hashes else None
        except Exception as exc:
            logger.warning(
                "Could not compute video frame hashes for %s: %s", path, exc
            )
            return None
        finally:
            cap.release()

    def _sample_positions(self, duration_seconds: float) -> list[float]:
        """Return the seconds at which to sample keyframes."""
        if duration_seconds <= 0:
            return []
        if self.frame_count == 1:
            return [duration_seconds / 2.0]
        # Avoid the very first and last frames to skip intros/fades.
        margin = duration_seconds * 0.1
        usable = max(duration_seconds - 2 * margin, 0.0)
        if usable <= 0:
            return [duration_seconds / 2.0]
        step = usable / (self.frame_count - 1)
        return [margin + i * step for i in range(self.frame_count)]

    def distance(self, hash_a: str, hash_b: str) -> int:
        """Return the Hamming distance between two pHash hex strings."""
        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)  # type: ignore[attr-defined]
