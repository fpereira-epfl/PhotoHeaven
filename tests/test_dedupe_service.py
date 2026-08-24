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
        self._video_frame_hashes: dict[str, list[str]] = {}

    def list_media(self, limit: int = 100, offset: int = 0) -> list[MediaFile]:
        return self._items[offset : offset + limit]

    def update_media_perceptual_hash(
        self, media_id: str, perceptual_hash: str
    ) -> None:
        self._perceptual[media_id] = perceptual_hash
        for item in self._items:
            if item.id == media_id:
                item.perceptual_hash = perceptual_hash

    def update_media_video_frame_hashes(
        self, media_id: str, frame_hashes: list[str]
    ) -> None:
        self._video_frame_hashes[media_id] = frame_hashes
        for item in self._items:
            if item.id == media_id:
                item.video_frame_hashes = frame_hashes

    def update_media_path(self, media_id: str, new_path: str) -> None:
        for item in self._items:
            if item.id == media_id:
                item.path = new_path

    def get_media_id_by_path(self, path: str) -> str | None:
        for item in self._items:
            if item.path == path:
                return item.id
        return None

    def clear_duplicate_groups(self) -> None:
        self._groups = []

    def save_duplicate_group(
        self, group_id: str, members: list[dict]
    ) -> None:
        self._groups.append({"group_id": group_id, "members": members})

    def list_duplicate_groups(self) -> list[dict]:
        return [
            {
                "group_id": g["group_id"],
                "created_at": None,
                "members": [
                    {
                        "media_id": m["media_id"],
                        "path": item.path,
                        "size_bytes": item.size_bytes,
                        "checksum": item.checksum,
                        "is_primary": m.get("is_primary", False),
                        "match_level": m["match_level"],
                    }
                    for m in g["members"]
                    for item in self._items
                    if item.id == m["media_id"]
                ],
            }
            for g in self._groups
        ]


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


def test_metadata_similarity_does_not_create_false_positives(tmp_path: Path) -> None:
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

    assert result.groups_created == 0


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


def test_move_duplicates_organizes_into_year_month_folders(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    source = tmp_path / "files" / "2024" / "05" / "source.jpg"
    dup1 = tmp_path / "files" / "2024" / "05" / "dup1.jpg"
    dup2 = tmp_path / "files" / "2024" / "05" / "dup2.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (50, 50), color="black").save(source)
    shutil.copy2(source, dup1)
    shutil.copy2(source, dup2)

    dt = datetime(2024, 5, 1, 12, 0, 0)
    repo.save_media(
        _media(source, checksum="c1", size_bytes=5000, capture_datetime=dt)
    )
    repo.save_media(_media(dup1, checksum="c2", size_bytes=1000, capture_datetime=dt))
    repo.save_media(_media(dup2, checksum="c3", size_bytes=900, capture_datetime=dt))

    service = DedupeService(repo, PerceptualHasher())
    service.find_duplicates()

    duplicates_root = tmp_path / "duplicates"
    move_result = service.move_duplicates(duplicates_root)

    assert move_result.groups_processed == 1
    assert move_result.files_moved == 2
    assert move_result.groups_with_errors == 0

    expected_dup1 = duplicates_root / "2024" / "05" / "dup1.jpg"
    expected_dup2 = duplicates_root / "2024" / "05" / "dup2.jpg"
    assert expected_dup1.exists()
    assert expected_dup2.exists()
    assert not dup1.exists()
    assert not dup2.exists()
    assert source.exists()

    # Database paths should reflect the move.
    assert repo.get_by_checksum("c2").path == str(expected_dup1)
    assert repo.get_by_checksum("c3").path == str(expected_dup2)


def test_move_duplicates_dry_run_does_not_move_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    source = tmp_path / "files" / "2024" / "05" / "source.jpg"
    dup = tmp_path / "files" / "2024" / "05" / "dup.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (50, 50), color="black").save(source)
    shutil.copy2(source, dup)

    dt = datetime(2024, 5, 1, 12, 0, 0)
    repo.save_media(_media(source, checksum="c1", size_bytes=5000, capture_datetime=dt))
    repo.save_media(_media(dup, checksum="c2", size_bytes=1000, capture_datetime=dt))

    service = DedupeService(repo, PerceptualHasher())
    service.find_duplicates()

    move_result = service.move_duplicates(tmp_path / "duplicates", dry_run=True)

    assert move_result.files_moved == 1
    assert dup.exists()
    assert repo.get_by_checksum("c2").path == str(dup)


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


def test_list_duplicate_groups_only_faces_filters_correctly() -> None:
    a = MediaFile(
        id="m1",
        path="/a.jpg",
        checksum="same1",
        size_bytes=1000,
        mtime=1.0,
        media_type=MediaType.IMAGE,
    )
    b = MediaFile(
        id="m2",
        path="/b.jpg",
        checksum="same1",
        size_bytes=1000,
        mtime=1.0,
        media_type=MediaType.IMAGE,
    )
    c = MediaFile(
        id="m3",
        path="/c.jpg",
        checksum="same2",
        size_bytes=1000,
        mtime=1.0,
        media_type=MediaType.IMAGE,
    )
    d = MediaFile(
        id="m4",
        path="/d.jpg",
        checksum="same2",
        size_bytes=1000,
        mtime=1.0,
        media_type=MediaType.IMAGE,
    )

    class _Repo(_FakeDedupeRepository):
        def __init__(self, items, faces_media_ids):
            super().__init__(items)
            self._faces_media_ids = set(faces_media_ids)

        def get_media_ids_with_faces(self) -> set[str]:
            return self._faces_media_ids

        def list_duplicate_groups(self) -> list[dict]:
            return [
                {
                    "group_id": g["group_id"],
                    "created_at": None,
                    "members": [
                        {
                        "media_id": m["media_id"],
                        "path": item.path,
                        "size_bytes": item.size_bytes,
                        "is_primary": m.get("is_primary", False),
                        "match_level": m["match_level"],
                        }
                            for m in g["members"]
                            for item in self._items
                            if item.id == m["media_id"]
                    ],
                }
                for g in self._groups
            ]

    repo = _Repo([a, b, c, d], ["m3", "m4"])
    service = DedupeService(repo, PerceptualHasher())
    service.find_duplicates()

    all_groups = service.list_duplicate_groups()
    face_groups = service.list_duplicate_groups(only_faces=True)

    assert len(all_groups) == 2
    assert len(face_groups) == 1
    member_ids = {m["media_id"] for m in face_groups[0]["members"]}
    assert member_ids == {"m3", "m4"}
