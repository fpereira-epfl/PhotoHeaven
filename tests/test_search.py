"""Tests for media search."""

from datetime import datetime
from pathlib import Path

import pytest

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.ports import MediaSearchQuery
from photoheaven.domain.models import Face, MediaFile, MediaType


@pytest.fixture
def repository(tmp_path: Path) -> SqliteMediaRepository:
    db = tmp_path / "test.db"
    return SqliteMediaRepository(str(db))


def _media(
    path: str,
    *,
    media_type: MediaType = MediaType.IMAGE,
    capture_datetime: datetime | None = None,
    size_bytes: int = 100,
) -> MediaFile:
    return MediaFile(
        id=f"id-{Path(path).name}",
        path=path,
        checksum=f"hash-{Path(path).name}",
        size_bytes=size_bytes,
        mtime=1234567890.0,
        media_type=media_type,
        capture_datetime=capture_datetime,
    )


def _face(media_id: str, name: str) -> Face:
    return Face(
        id=f"face-{media_id}-{name}",
        media_id=media_id,
        bbox=(0, 0, 10, 10),
        embedding=[0.0] * 512,
        identity_name=name,
    )


def test_search_by_year_and_month(repository: SqliteMediaRepository) -> None:
    may = _media("/files/may.jpg", capture_datetime=datetime(2020, 5, 15, 10, 0))
    june = _media("/files/june.jpg", capture_datetime=datetime(2020, 6, 1, 10, 0))
    other = _media("/files/other.jpg", capture_datetime=datetime(2019, 5, 15, 10, 0))
    for m in (may, june, other):
        repository.save_media(m)

    results = repository.search_media(
        MediaSearchQuery(year=2020, month=5)
    )
    assert len(results) == 1
    assert results[0].path == "/files/may.jpg"


def test_search_by_date_range(repository: SqliteMediaRepository) -> None:
    early = _media("/files/early.jpg", capture_datetime=datetime(2020, 1, 31, 10, 0))
    mid = _media("/files/mid.jpg", capture_datetime=datetime(2020, 3, 15, 10, 0))
    late = _media("/files/late.jpg", capture_datetime=datetime(2020, 6, 1, 10, 0))
    for m in (early, mid, late):
        repository.save_media(m)

    results = repository.search_media(
        MediaSearchQuery(
            date_from=datetime(2020, 2, 1),
            date_to=datetime(2020, 5, 31, 23, 59, 59),
        )
    )
    paths = {m.path for m in results}
    assert paths == {"/files/mid.jpg"}


def test_search_by_name(repository: SqliteMediaRepository) -> None:
    alice_pic = _media("/files/alice.jpg")
    bob_pic = _media("/files/bob.jpg")
    nobody = _media("/files/nobody.jpg")
    for m in (alice_pic, bob_pic, nobody):
        repository.save_media(m)

    repository.save_face(_face(alice_pic.id, "Alice"))
    repository.save_face(_face(bob_pic.id, "Bob"))

    results = repository.search_media(
        MediaSearchQuery(names=["Alice"])
    )
    assert len(results) == 1
    assert results[0].path == "/files/alice.jpg"


def test_search_excludes_videos_by_default(repository: SqliteMediaRepository) -> None:
    img = _media("/files/photo.jpg", capture_datetime=datetime(2020, 1, 1))
    vid = _media(
        "/files/video.mov",
        media_type=MediaType.VIDEO,
        capture_datetime=datetime(2020, 1, 1),
    )
    repository.save_media(img)
    repository.save_media(vid)

    results = repository.search_media(MediaSearchQuery(year=2020))
    assert len(results) == 1
    assert results[0].media_type == MediaType.IMAGE


def test_search_includes_videos_when_requested(
    repository: SqliteMediaRepository,
) -> None:
    img = _media("/files/photo.jpg", capture_datetime=datetime(2020, 1, 1))
    vid = _media(
        "/files/video.mov",
        media_type=MediaType.VIDEO,
        capture_datetime=datetime(2020, 1, 1),
    )
    repository.save_media(img)
    repository.save_media(vid)

    results = repository.search_media(
        MediaSearchQuery(year=2020, include_videos=True)
    )
    assert len(results) == 2


def test_search_limit(repository: SqliteMediaRepository) -> None:
    for i in range(5):
        repository.save_media(
            _media(f"/files/{i}.jpg", capture_datetime=datetime(2020, 1, i + 1))
        )

    results = repository.search_media(MediaSearchQuery(year=2020, limit=2))
    assert len(results) == 2
