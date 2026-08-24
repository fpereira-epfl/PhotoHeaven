"""Perceptual hashing adapter for videos via sampled keyframes."""

from __future__ import annotations

import contextlib
import logging
import os
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


try:
    from pymediainfo import MediaInfo

    _HAVE_MEDIAINFO = True
except Exception:  # pragma: no cover
    _HAVE_MEDIAINFO = False


@contextlib.contextmanager
def _silence_ffmpeg_stderr():
    """Redirect the C-level stderr to /dev/null while OpenCV talks to FFmpeg."""
    stderr_fileno = 2
    old_stderr = os.dup(stderr_fileno)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fileno)
        yield
    finally:
        os.dup2(old_stderr, stderr_fileno)
        os.close(old_stderr)
        os.close(devnull)


def _duration_from_mediainfo(path: Path) -> float | None:
    """Return duration in seconds from container metadata, if available."""
    if not _HAVE_MEDIAINFO:
        return None
    try:
        media_info = MediaInfo.parse(str(path))
        for track in media_info.tracks:
            duration = getattr(track, "duration", None)
            if duration:
                return float(duration) / 1000.0
    except Exception as exc:
        logger.debug("Could not read MediaInfo duration for %s: %s", path, exc)
    return None


class VideoFrameHashResult:
    """Result of computing perceptual hashes for sampled video keyframes."""

    def __init__(
        self,
        frame_hashes: list[str],
        duration_seconds: float | None = None,
    ) -> None:
        self.frame_hashes = frame_hashes
        self.duration_seconds = duration_seconds


class VideoFrameHasher:
    """Compute pHashes for a small set of keyframes extracted from a video."""

    def __init__(self, frame_count: int = 3) -> None:
        if not _VIDEO_HASH_AVAILABLE:
            raise RuntimeError(
                "opencv-python, imagehash and Pillow are required for video hashing"
            )
        # Silence OpenCV's own Python-level logging. FFmpeg warnings are silenced
        # via stderr redirection while decoding.
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # type: ignore[attr-defined]
        self.frame_count = frame_count

    def compute(self, path: Path) -> VideoFrameHashResult | None:
        """Return pHashes and duration for sampled keyframes, or None on failure."""
        try:
            with _silence_ffmpeg_stderr():
                cap = cv2.VideoCapture(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Could not open video %s: %s", path, exc)
            return None

        if not cap.isOpened():
            logger.warning("Could not open video %s", path)
            return None

        try:
            with _silence_ffmpeg_stderr():
                # Sanity-check the stream by reading the first frame. Corrupted
                # files often open successfully but fail immediately on decode.
                ok, _ = cap.read()
                if not ok:
                    logger.warning("Could not decode first frame of %s", path)
                    return None

                # Reset before querying properties; some containers report more
                # accurate metadata after a successful read.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore[attr-defined]

                frame_count_total = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)  # type: ignore[attr-defined]
                )
                fps = cap.get(cv2.CAP_PROP_FPS)  # type: ignore[attr-defined]
                duration_seconds: float | None = None
                if fps:
                    duration_seconds = frame_count_total / fps
                if not duration_seconds or duration_seconds <= 0:
                    duration_seconds = _duration_from_mediainfo(path)

                if not duration_seconds or duration_seconds <= 0:
                    logger.warning("Could not determine video duration for %s", path)
                    return None

                positions = self._sample_positions(duration_seconds)
                hashes: list[str] = []
                for position in positions:
                    cap.set(  # type: ignore[attr-defined]
                        cv2.CAP_PROP_POS_MSEC,  # type: ignore[attr-defined]
                        position * 1000.0,
                    )
                    success, frame = cap.read()
                    if not success:
                        continue
                    try:
                        image = Image.fromarray(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
                        )
                        hashes.append(str(imagehash.phash(image)))
                    except Exception as exc:
                        logger.debug(
                            "Could not hash keyframe at %.1fs for %s: %s",
                            position,
                            path,
                            exc,
                        )
                if not hashes:
                    logger.warning(
                        "Could not extract any hashable keyframes from %s", path
                    )
                    return None
                return VideoFrameHashResult(
                    frame_hashes=hashes,
                    duration_seconds=duration_seconds,
                )
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
