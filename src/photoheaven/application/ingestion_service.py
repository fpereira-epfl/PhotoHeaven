"""Application service that orchestrates media ingestion."""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from photoheaven.application.ports import (
    Hasher,
    MediaRepository,
    MetadataExtractor,
)
from photoheaven.domain.models import MediaFile, MediaType

logger = logging.getLogger(__name__)

_IMAGE_MIME_PREFIXES = {"image/"}
_VIDEO_MIME_PREFIXES = {"video/", "application/mp4"}


def guess_media_type(path: Path) -> MediaType:
    """Guess image vs video from extension/mimetype."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if any(mime.startswith(prefix) for prefix in _IMAGE_MIME_PREFIXES):
            return MediaType.IMAGE
        if any(mime.startswith(prefix) for prefix in _VIDEO_MIME_PREFIXES):
            return MediaType.VIDEO
    # Fallback to common extensions.
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".heic", ".webp", ".avif", ".dng"}:
        return MediaType.IMAGE
    if suffix in {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
        ".3gp",
        ".webm",
        ".mts",
        ".m2ts",
        ".ts",
        ".mpg",
        ".mpeg",
    }:
        return MediaType.VIDEO
    return MediaType.UNKNOWN


@dataclass(frozen=True)
class IngestResult:
    status: str  # added, updated, skipped, error
    media: MediaFile | None = None
    message: str = ""
    metadata_extracted: bool = True
    """False when the metadata extractor raised an exception (e.g. corrupt image)."""


class IngestionService:
    """Ingest a media file into the library, skipping unchanged files."""

    def __init__(
        self,
        hasher: Hasher,
        metadata_extractor: MetadataExtractor,
        repository: MediaRepository,
    ) -> None:
        self.hasher = hasher
        self.metadata_extractor = metadata_extractor
        self.repository = repository

    def check_metadata(self, path: Path) -> bool:
        """Return True if metadata can be extracted from *path*.

        This performs a lightweight re-check without writing to the repository.
        It is used by the CLI when moving corrupted files from an already
        ingested library.
        """
        media_type = guess_media_type(path)
        try:
            self.metadata_extractor.extract(path, media_type)
            return True
        except Exception:
            logger.exception("Metadata extraction failed for %s", path)
            return False

    def ingest_file(self, path: Path, *, force: bool = False) -> IngestResult:
        if not path.is_file():
            return IngestResult(
                status="error", message=f"Not a file: {path}"
            )

        try:
            checksum = self.hasher.hash_file(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Checksum failed for %s", path)
            return IngestResult(
                status="error", message=f"Checksum failed: {exc}"
            )

        stat = path.stat()
        existing = self.repository.get_by_checksum(checksum)

        if existing is not None and not force:
            if existing.mtime == stat.st_mtime:
                return IngestResult(
                    status="skipped",
                    media=existing,
                    message=f"Already ingested: {path}",
                    metadata_extracted=True,
                )

        media_type = guess_media_type(path)
        try:
            metadata = self.metadata_extractor.extract(path, media_type)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Metadata extraction failed for %s", path)
            # Continue with whatever we know; do not fail the whole ingestion.
            metadata = None

        if metadata is not None:
            metadata_extracted = metadata.extracted
            if metadata.media_type is not MediaType.UNKNOWN:
                media_type = metadata.media_type
            capture_datetime = metadata.capture_datetime
            make = metadata.make
            model = metadata.model
            gps = metadata.gps
        else:
            metadata_extracted = False
            capture_datetime = None
            make = None
            model = None
            gps = None

        if existing is not None:
            media = MediaFile(
                id=existing.id,
                path=str(path.resolve()),
                checksum=checksum,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                media_type=media_type,
                capture_datetime=capture_datetime,
                make=make,
                model=model,
                gps=gps,
                face_analysis_at=existing.face_analysis_at,
                face_analysis_version=existing.face_analysis_version,
                metadata_extracted=metadata_extracted,
                perceptual_hash=existing.perceptual_hash,
                created_at=existing.created_at,
                updated_at=datetime.utcnow(),
            )
        else:
            media = MediaFile(
                id=str(uuid4()),
                path=str(path.resolve()),
                checksum=checksum,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                media_type=media_type,
                capture_datetime=capture_datetime,
                make=make,
                model=model,
                gps=gps,
                metadata_extracted=metadata_extracted,
            )

        self.repository.save_media(media)
        status = "updated" if existing is not None else "added"
        return IngestResult(
            status=status,
            media=media,
            message=f"{status.capitalize()}: {path}",
            metadata_extracted=metadata_extracted,
        )
