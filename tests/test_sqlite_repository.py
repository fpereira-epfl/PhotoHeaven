"""Tests for SQLite repository persistence behaviour."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.domain.models import Face, GeoPoint, MediaFile, MediaType


def _media(path: str, checksum: str | None = None) -> MediaFile:
    return MediaFile(
        id=str(uuid4()),
        path=path,
        checksum=checksum or str(uuid4()),
        size_bytes=1000,
        mtime=1234567890.0,
        media_type=MediaType.IMAGE,
        capture_datetime=datetime.utcnow(),
    )


def _face(media_id: str) -> Face:
    return Face(
        id=str(uuid4()),
        media_id=media_id,
        bbox=(10, 20, 30, 40),
        embedding=[0.0] * 512,
        embedding_version="test_v1",
        detection_confidence=0.9,
    )


def test_update_preserves_faces(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _media("/photos/IMG_1.jpg")
    repo.save_media(media)
    repo.save_face(_face(media.id))

    # Simulate a re-ingest that updates metadata but keeps the same id.
    media.capture_datetime = datetime(2020, 1, 1, 12, 0, 0)
    media.updated_at = datetime.utcnow()
    repo.save_media(media)

    reloaded = repo.get_by_checksum(media.checksum)
    assert reloaded is not None
    assert reloaded.path == "/photos/IMG_1.jpg"
    assert reloaded.capture_datetime == datetime(2020, 1, 1, 12, 0, 0)
    assert repo.media_has_faces(media.id)
    assert repo.count_faces() == 1


def test_rename_updates_path_and_preserves_faces(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _media("/photos/old_name.jpg")
    repo.save_media(media)
    repo.save_face(_face(media.id))

    # Simulate moving/renaming the file and re-ingesting.
    media.path = "/photos/new_name.jpg"
    media.updated_at = datetime.utcnow()
    repo.save_media(media)

    reloaded = repo.get_by_checksum(media.checksum)
    assert reloaded is not None
    assert reloaded.path == "/photos/new_name.jpg"
    assert repo.media_has_faces(media.id)
    assert repo.count_faces() == 1


def test_content_change_at_same_path_replaces_record(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    old_media = _media("/photos/IMG_1.jpg", checksum="old_checksum")
    repo.save_media(old_media)
    repo.save_face(_face(old_media.id))

    # A different file now exists at the same path.
    new_media = _media("/photos/IMG_1.jpg", checksum="new_checksum")
    new_media.capture_datetime = datetime(2021, 1, 1, 12, 0, 0)
    repo.save_media(new_media)

    # Old record and its faces should be gone.
    assert repo.get_by_checksum("old_checksum") is None
    assert repo.media_has_faces(old_media.id) is False

    # New record should exist and be queryable by checksum.
    reloaded = repo.get_by_checksum("new_checksum")
    assert reloaded is not None
    assert reloaded.id == new_media.id
    assert reloaded.path == "/photos/IMG_1.jpg"
    assert repo.count_faces() == 0
