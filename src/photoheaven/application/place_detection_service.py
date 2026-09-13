"""Application service that orchestrates place/scene detection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from photoheaven.application.ports import MediaRepository, PlaceIdentifier
from photoheaven.domain.models import MediaFile, MediaType, PlaceRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaceDetectionResult:
    """Result summary for a place-detection run."""

    processed: int = 0
    skipped: int = 0
    errors: int = 0
    countries: int = 0
    scenes: int = 0
    records: list[PlaceRecord] = field(default_factory=list)


class PlaceDetectionService:
    """Run country/scene detection on library media and persist the results.

    The service depends only on the ``MediaRepository`` port and the
    ``PlaceIdentifier`` port, keeping it independent of any particular ML
    library.
    """

    def __init__(
        self,
        identifier: PlaceIdentifier,
        repository: MediaRepository,
    ) -> None:
        self.identifier = identifier
        self.repository = repository

    def detect(
        self,
        *,
        batch_size: int = 100,
        limit: int | None = None,
        random: bool = False,
        force: bool = False,
        progress_callback: Callable[[MediaFile, PlaceRecord], None] | None = None,
    ) -> PlaceDetectionResult:
        """Detect places on eligible, unprocessed media files.

        Args:
            batch_size: Number of media files to fetch per database query.
            limit: Maximum total number of media files to consider. When set,
                the service stops after this many items have been fetched from
                the repository, regardless of whether they were processed or
                skipped.
            random: Pick media files at random. Must be used together with
                ``limit``.
            force: Re-analyse media even if place analysis has already run.
            progress_callback: Optional callback invoked after each successful
                analysis with the media file and its place record.

        Returns:
            A summary of processed, skipped, errored, country, and scene counts.
        """
        if random and limit is None:
            raise ValueError("random selection requires a limit")

        result = PlaceDetectionResult()
        considered = 0

        if random:
            target_media = self.repository.list_media_random(limit=limit)
            for media in target_media:
                considered += 1
                file_result = self._process_media(
                    media, force=force, progress_callback=progress_callback
                )
                result = PlaceDetectionResult(
                    processed=result.processed + file_result.processed,
                    skipped=result.skipped + file_result.skipped,
                    errors=result.errors + file_result.errors,
                    countries=result.countries + file_result.countries,
                    scenes=result.scenes + file_result.scenes,
                    records=result.records + file_result.records,
                )
            return result

        offset = 0
        while True:
            media_batch = self.repository.list_media(
                limit=batch_size, offset=offset
            )

            if not media_batch:
                break

            for media in media_batch:
                if limit is not None and considered >= limit:
                    break
                considered += 1

                file_result = self._process_media(
                    media, force=force, progress_callback=progress_callback
                )
                result = PlaceDetectionResult(
                    processed=result.processed + file_result.processed,
                    skipped=result.skipped + file_result.skipped,
                    errors=result.errors + file_result.errors,
                    countries=result.countries + file_result.countries,
                    scenes=result.scenes + file_result.scenes,
                    records=result.records + file_result.records,
                )

            if limit is not None and considered >= limit:
                break
            if len(media_batch) < batch_size:
                break
            offset += batch_size

        return result

    def _process_media(
        self,
        media: MediaFile,
        *,
        force: bool,
        progress_callback: Callable[[MediaFile, PlaceRecord], None] | None,
    ) -> PlaceDetectionResult:
        """Analyse a single media file and persist the place record."""
        if not force and media.place_analysis_at is not None:
            logger.debug("Skipping already-analysed media: %s", media.path)
            return PlaceDetectionResult(skipped=1)

        if media.media_type is not MediaType.IMAGE:
            logger.debug("Skipping non-image media: %s", media.path)
            return PlaceDetectionResult(skipped=1)

        if not Path(media.path).exists():
            logger.warning(
                "Media file no longer exists on disk: %s. "
                "Run 'ph sync --prune' to remove stale records.",
                media.path,
            )
            return PlaceDetectionResult(errors=1)

        try:
            record = self.identifier.identify(media)
        except Exception as exc:
            if isinstance(exc, FileNotFoundError):
                logger.warning(
                    "Media file disappeared during analysis: %s", media.path
                )
            else:
                logger.exception("Place analysis failed for %s", media.path)
            return PlaceDetectionResult(errors=1)

        try:
            self.repository.save_place_record(record)
        except Exception:
            logger.exception("Failed to save place record for %s", media.path)
            return PlaceDetectionResult(processed=1, errors=1)

        analyzed_at = datetime.utcnow()
        try:
            self.repository.update_media_place_analysis(
                media_id=media.id,
                analyzed_at=analyzed_at,
                version=self.identifier.version,
            )
        except Exception:
            logger.exception("Failed to mark media as analysed: %s", media.path)
            return PlaceDetectionResult(processed=1, errors=1)

        logger.info(
            "Identified place for %s: country=%s scene=%s",
            media.path,
            record.country or "—",
            record.scene or "—",
        )
        if progress_callback is not None:
            progress_callback(media, record)
        return PlaceDetectionResult(
            processed=1,
            countries=1 if record.country else 0,
            scenes=1 if record.scene else 0,
            records=[record],
        )
