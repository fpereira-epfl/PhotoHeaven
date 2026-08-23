"""Tests for the deduplication service."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from photoheaven.adapters.integrity.perceptual_hasher import PerceptualHasher
from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.dedupe_service import DedupeService
from photoheaven.domain.models import MediaFile, MediaType


def _media(
    path: Path,
    *,
    checksum: str | None = None,
    size_bytes: int | None = None,
    capture_datetime: datetime | None = None,
    make: str | None = None,
    model: str | None = None,
    media_type: MediaType = MediaType.IMAGE,
    perceptual_hash: str | None = None,
) -> MediaFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        Image.new("RGB", (10, 10), color="red").save(path)
    return MediaFile(
        id=str(uuid4()),
        path=str(path),
        checksum=checksum or str(uuid4()),
        size_bytes=size_bytes if size_bytes is not None else path.stat().st_size,
        mtime=path.stat().st_mtime,
        media_type=media_type,
        capture_datetime=capture_datetime,
        make=make,
        model=model,
        perceptual_hash=perceptual_hash,
    )


def _repo(tmp_path: Path) -> SqliteMediaRepository:
    db_path = tmp_path / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    return SqliteMediaRepository(str(db_path))


class _FakeDedupeRepository:
    """Minimal repository for dedupe tests that need duplicate checksums."""

    def __init__(self, items: list[MediaFile]) -> None:
        self._items = items
        self._groups: list[dict] = []
        self._perceptual: dict[str, str] = {}

    def list_media(self, limit: int = 100, offset: int = 0) -> list[MediaFile]:
        return self._items[offset : offset + limit]

    def update_media_perceptual_hash(
        self, media_id: str, perceptual_hash: str
    ) -> None:
        self._perceptual[media_id] = perceptual_hash
        for item in self._items:
            if item.id == media_id:
                item.perceptual_hash = perceptual_hash

    def clear_duplicate_groups(self) -> None:
        self._groups = []

    def save_duplicate_group(
        self, group_id: str, members: list[dict]
    ) -> None:
        self._groups.append({"group_id": group_id, "members": members})

    def list_duplicate_groups(self) -> list[dict]:
        return self._groups


def test_finds_checksum_duplicates() -> None:
    a = MediaFile(
        id="m1",
        path="/a.jpg",
        checksum="same",
        size_bytes=1000,
        mtime=1.0,
        media_type=MediaType.IMAGE,
    )
    b = MediaFile(
        id="m2",
        path="/b.jpg",
        checksum="same",
        size_bytes=1000,
        mtime=1.0,
        media_type=MediaType.IMAGE,
    )
    repo = _FakeDedupeRepository([a, b])
    service = DedupeService(repo, PerceptualHasher())
    result = service.find_duplicates()

    assert result.groups_created == 1
    assert result.total_duplicates == 1
    assert result.checksum_matches == 1


def test_progress_callback_reports_totals_and_groups() -> None:
    a = MediaFile(
        id="m1",
        path="/a.jpg",
        checksum="same",
        size_bytes=1000,
        mtime=1.0,
        media_type=MediaType.IMAGE,
    )
    b = MediaFile(
        id="m2",
        path="/b.jpg",
        checksum="same",
        size_bytes=1000,
        mtime=1.0,
        media_type=MediaType.IMAGE,
    )
    repo = _FakeDedupeRepository([a, b])
    service = DedupeService(repo, PerceptualHasher())

    snapshots: list[Any] = []

    def callback(progress: Any) -> None:
        snapshots.append(progress)

    service.find_duplicates(progress_callback=callback)

    assert snapshots
    first = snapshots[0]
    assert first.total_media == 2
    last = snapshots[-1]
    assert last.groups_found == 1
    assert last.duplicate_files_found == 1


def test_finds_perceptual_duplicates_across_formats(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    jpeg = tmp_path / "a.jpg"
    png = tmp_path / "a.png"
    Image.new("RGB", (50, 50), color="green").save(jpeg, "JPEG")
    Image.new("RGB", (50, 50), color="green").save(png, "PNG")

    dt = datetime(2024, 5, 1, 12, 0, 0)
    repo.save_media(_media(jpeg, capture_datetime=dt))
    repo.save_media(_media(png, capture_datetime=dt))

    service = DedupeService(repo, PerceptualHasher())
    result = service.find_duplicates()

    assert result.groups_created == 1
    assert result.perceptual_matches >= 1


def test_finds_metadata_duplicates(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    # Visually different images so perceptual hashes will not match at distance 0.
    Image.effect_noise((100, 100), 10).convert("RGB").save(a)
    Image.effect_noise((100, 100), 50).convert("RGB").save(b)

    dt = datetime(2024, 5, 1, 12, 0, 0)
    repo.save_media(
        _media(a, checksum="c1", size_bytes=1000, capture_datetime=dt, make="Apple", model="iPhone")
    )
    repo.save_media(
        _media(b, checksum="c2", size_bytes=1005, capture_datetime=dt, make="Apple", model="iPhone")
    )

    service = DedupeService(repo, PerceptualHasher())
    result = service.find_duplicates(max_distance=0)

    assert result.groups_created == 1
    assert result.metadata_matches == 1


def test_prefers_heic_as_primary(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    heic = tmp_path / "a.heic"
    jpeg = tmp_path / "a.jpg"
    Image.new("RGB", (50, 50), color="yellow").save(heic, "JPEG")
    Image.new("RGB", (50, 50), color="yellow").save(jpeg, "JPEG")

    dt = datetime(2024, 5, 1, 12, 0, 0)
    # Make the JPEG larger; HEIC should still win on format priority.
    repo.save_media(_media(heic, size_bytes=1000, capture_datetime=dt))
    repo.save_media(_media(jpeg, size_bytes=5000, capture_datetime=dt))

    service = DedupeService(repo, PerceptualHasher())
    service.find_duplicates()
    groups = service.list_duplicate_groups()

    assert len(groups) == 1
    members = sorted(groups[0]["members"], key=lambda m: m["is_primary"], reverse=True)
    assert members[0]["path"].endswith(".heic")


def test_list_duplicate_groups_sorted_by_primary_quality(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    source = tmp_path / "source.jpg"
    Image.new("RGB", (50, 50), color="black").save(source)

    large = tmp_path / "large.jpg"
    small = tmp_path / "small.jpg"
    shutil.copy2(source, large)
    shutil.copy2(source, small)

    dt = datetime(2024, 5, 1, 12, 0, 0)
    repo.save_media(_media(large, checksum="c1", size_bytes=5000, capture_datetime=dt))
    repo.save_media(_media(small, checksum="c2", size_bytes=1000, capture_datetime=dt))

    # A separate group with a smaller primary and a different datetime.
    other = tmp_path / "other.jpg"
    other2 = tmp_path / "other2.jpg"
    Image.effect_noise((50, 50), 30).convert("RGB").save(other)
    shutil.copy2(other, other2)
    other_dt = datetime(2023, 1, 1, 10, 0, 0)
    repo.save_media(_media(other, checksum="c3", size_bytes=300, capture_datetime=other_dt))
    repo.save_media(_media(other2, checksum="c4", size_bytes=300, capture_datetime=other_dt))

    service = DedupeService(repo, PerceptualHasher())
    service.find_duplicates()
    groups = service.list_duplicate_groups()

    assert len(groups) == 2
    assert groups[0]["members"][0]["size_bytes"] == 5000
