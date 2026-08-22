"""InsightFace-based face detection and embedding adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from photoheaven.application.ports import FaceAnalyzer
from photoheaven.domain.models import Face, MediaFile, MediaType

logger = logging.getLogger(__name__)


def _default_providers() -> list[str]:
    """Pick a reasonable ONNX execution provider for the current machine."""
    try:
        import onnxruntime as ort
    except Exception:  # pragma: no cover
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    for provider in preferred:
        if provider in available:
            return [provider]
    return ["CPUExecutionProvider"]


class InsightFaceAnalyzer(FaceAnalyzer):
    """Detect faces and compute ArcFace embeddings with InsightFace.

    The first run downloads the detection/recognition models from the internet
    into ``~/.insightface/models``.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
    ) -> None:
        try:
            import cv2
            import insightface
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "InsightFace dependencies are not installed. "
                "Install the 'face' extra (insightface, onnxruntime, opencv-python)."
            ) from exc

        self._cv2 = cv2
        self._np = np
        providers = providers or _default_providers()
        logger.info("Initializing InsightFace with providers=%s", providers)
        self.app = insightface.app.FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        self._embedding_version = f"insightface_{model_name}_arcface"

    def analyze(self, media: MediaFile) -> list[Face]:
        if media.media_type is not MediaType.IMAGE:
            logger.debug("Skipping face analysis for non-image file: %s", media.path)
            return []

        path = Path(media.path)
        if not path.is_file():
            logger.warning("File not found for face analysis: %s", media.path)
            return []

        try:
            img = self._cv2.imread(str(path))
            if img is None:
                logger.warning("OpenCV could not read image: %s", media.path)
                return []
            detected = self.app.get(img)
        except Exception as exc:
            logger.exception("Face analysis failed for %s", media.path)
            return []

        faces: list[Face] = []
        for det in detected:
            bbox = tuple(int(round(v)) for v in det.bbox.flatten().tolist())
            embedding = det.embedding.flatten().tolist()
            faces.append(
                Face(
                    id=str(uuid4()),
                    media_id=media.id,
                    bbox=bbox,
                    embedding=embedding,
                    embedding_version=self._embedding_version,
                    detection_confidence=float(det.det_score)
                    if hasattr(det, "det_score")
                    else 0.0,
                )
            )

        logger.info("Detected %d face(s) in %s", len(faces), media.path)
        return faces
