"""Application service that orchestrates face detection across the library."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from photoheaven.application.ports import FaceAnalyzer, MediaRepository
from photoheaven.domain.models import MediaFile, MediaType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceDetectionResult:
    """Result summary for a face-detection run."""

    processed: int = 0
    skipped: int = 0
    errors: int = 0
    faces_detected: int = 0


class FaceDetectionService:
    """Run face detection on library media and persist the results.

    The service is intentionally framework-agnostic: it depends only on the
    ``MediaRepository`` port and the ``FaceAnalyzer`` port.
    """

    def __init__(
        self,
        analyzer: FaceAnalyzer,
        repository: MediaRepository,
    ) -> None:
        self.analyzer = analyzer
        self.repository = repository

    def detect(
        self,
        *,
        batch_size: int = 100,
        force: bool = False,
        min_confidence: float = 0.0,
    ) -> FaceDetectionResult:
        """Detect faces on all eligible, unprocessed media files.

        Args:
            batch_size: Number of media files to fetch per database query.
            force: Re-analyse media even if face analysis has already run.
            min_confidence: Drop detected faces with a confidence score below
                this value.

        Returns:
            A summary of processed, skipped, errored, and detected-face counts.
        """
        result = FaceDetectionResult()
        offset = 0

        while True:
            media_batch = self.repository.list_media(
                limit=batch_size, offset=offset
            )

            if not media_batch:
                break

            for media in media_batch:
                file_result = self._process_media(
                    media, force=force, min_confidence=min_confidence
                )
                result = FaceDetectionResult(
                    processed=result.processed + file_result.processed,
                    skipped=result.skipped + file_result.skipped,
                    errors=result.errors + file_result.errors,
                    faces_detected=result.faces_detected
                    + file_result.faces_detected,
                )

            if len(media_batch) < batch_size:
                break
            offset += batch_size

        return result

    def _process_media(
        self,
        media: MediaFile,
        *,
        force: bool,
        min_confidence: float,
    ) -> FaceDetectionResult:
        """Analyse a single media file and persist any faces found."""
        if not force and media.face_analysis_at is not None:
            logger.debug("Skipping already-analysed media: %s", media.path)
            return FaceDetectionResult(skipped=1)

        if media.media_type is not MediaType.IMAGE:
            logger.debug("Skipping non-image media: %s", media.path)
            return FaceDetectionResult(skipped=1)

        try:
            faces = self.analyzer.analyze(media)
        except Exception:
            logger.exception("Face analysis failed for %s", media.path)
            return FaceDetectionResult(errors=1)

        accepted_faces = [
            face
            for face in faces
            if face.detection_confidence >= min_confidence
        ]

        for face in accepted_faces:
            try:
                self.repository.save_face(face)
            except Exception:
                logger.exception("Failed to save face for %s", media.path)
                return FaceDetectionResult(
                    processed=1,
                    errors=1,
                    faces_detected=len(accepted_faces),
                )

        analyzed_at = datetime.utcnow()
        try:
            self.repository.update_media_face_analysis(
                media_id=media.id,
                analyzed_at=analyzed_at,
                version=self.analyzer.version,
            )
        except Exception:
            logger.exception("Failed to mark media as analysed: %s", media.path)
            return FaceDetectionResult(processed=1, errors=1)

        logger.info(
            "Detected %d face(s) in %s",
            len(accepted_faces),
            media.path,
        )
        return FaceDetectionResult(
            processed=1, faces_detected=len(accepted_faces)
        )
