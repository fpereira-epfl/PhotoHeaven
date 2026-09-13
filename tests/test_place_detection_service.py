"""Tests for place/scene detection service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.place_detection_service import PlaceDetectionService
from photoheaven.application.ports import PlaceIdentifier
from photoheaven.domain.models import MediaFile, MediaType, PlaceRecord


class _FakePlaceIdentifier(PlaceIdentifier):
    """Test double that predicts based on the media path."""

    @property
    def version(self) -> str:
        return "fake-v1"

    def identify(self, media: MediaFile) -> PlaceRecord:
        return PlaceRecord(
            media_id=media.id,
            country="FakeCountry",
            country_source="visual",
            country_confidence=0.8,
            scene="beach",
            scene_confidence=0.7,
            version=self.version,
        )


def _media(path: str) -> MediaFile:
    return MediaFile(
        id=str(uuid4()),
        path=path,
        checksum=str(uuid4()),
        size_bytes=1000,
        mtime=1234567890.0,
        media_type=MediaType.IMAGE,
        capture_datetime=datetime.utcnow(),
    )


def _image_media(tmp_path: Path, filename: str) -> MediaFile:
    """Create a real image file and return its MediaFile."""
    image_path = tmp_path / filename
    Image.new("RGB", (8, 8), color="blue").save(image_path)
    media = _media(str(image_path))
    media.size_bytes = image_path.stat().st_size
    return media


def test_detect_skips_already_analysed_media(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _media(str(tmp_path / "photo.jpg"))
    repo.save_media(media)
    repo.update_media_place_analysis(
        media.id, analyzed_at=datetime.utcnow(), version="old-v1"
    )

    service = PlaceDetectionService(
        identifier=_FakePlaceIdentifier(), repository=repo
    )
    result = service.detect(batch_size=10)

    assert result.processed == 0
    assert result.skipped == 1
    assert repo.count_place_records() == 0


def test_detect_persists_place_records(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _image_media(tmp_path, "photo.jpg")
    repo.save_media(media)

    service = PlaceDetectionService(
        identifier=_FakePlaceIdentifier(), repository=repo
    )
    result = service.detect(batch_size=10)

    assert result.processed == 1
    assert result.countries == 1
    assert result.scenes == 1

    record = repo.get_place_record(media.id)
    assert record is not None
    assert record.country == "FakeCountry"
    assert record.scene == "beach"
    assert record.version == "fake-v1"

    reloaded_media = repo.get_by_path(media.path)
    assert reloaded_media is not None
    assert reloaded_media.place_analysis_version == "fake-v1"


def test_detect_respects_force(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _image_media(tmp_path, "photo.jpg")
    repo.save_media(media)
    repo.update_media_place_analysis(
        media.id, analyzed_at=datetime.utcnow(), version="old-v1"
    )

    service = PlaceDetectionService(
        identifier=_FakePlaceIdentifier(), repository=repo
    )
    result = service.detect(force=True, batch_size=10)

    assert result.processed == 1
    assert result.skipped == 0


def test_detect_counts_missing_files_as_errors(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _media(str(tmp_path / "missing.jpg"))
    repo.save_media(media)

    service = PlaceDetectionService(
        identifier=_FakePlaceIdentifier(), repository=repo
    )
    result = service.detect(batch_size=10)

    assert result.processed == 0
    assert result.errors == 1
    assert repo.count_place_records() == 0
    assert repo.get_place_record(media.id) is None


def test_detect_calls_progress_callback(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _image_media(tmp_path, "photo.jpg")
    repo.save_media(media)

    calls: list[tuple[str, str]] = []

    def _callback(media: MediaFile, record: PlaceRecord) -> None:
        calls.append((media.path, record.country or ""))

    service = PlaceDetectionService(
        identifier=_FakePlaceIdentifier(), repository=repo
    )
    service.detect(batch_size=10, progress_callback=_callback)

    assert len(calls) == 1
    assert calls[0][0] == str(tmp_path / "photo.jpg")
    assert calls[0][1] == "FakeCountry"


def test_detect_limit_stops_after_n_items(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    for idx in range(5):
        media = _image_media(tmp_path, f"photo_{idx}.jpg")
        repo.save_media(media)

    service = PlaceDetectionService(
        identifier=_FakePlaceIdentifier(), repository=repo
    )
    result = service.detect(batch_size=10, limit=2)

    assert result.processed == 2
    assert result.skipped == 0
    assert repo.count_place_records() == 2


def test_detect_random_requires_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))
    service = PlaceDetectionService(
        identifier=_FakePlaceIdentifier(), repository=repo
    )

    with pytest.raises(ValueError, match="random selection requires a limit"):
        service.detect(random=True)


def test_detect_random_selects_limited_items(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    for idx in range(10):
        media = _image_media(tmp_path, f"photo_{idx}.jpg")
        repo.save_media(media)

    service = PlaceDetectionService(
        identifier=_FakePlaceIdentifier(), repository=repo
    )
    result = service.detect(limit=3, random=True)

    assert result.processed == 3
    assert result.skipped == 0
    assert repo.count_place_records() == 3
