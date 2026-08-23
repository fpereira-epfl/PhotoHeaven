"""Tests for SQLite repository persistence behaviour."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.domain.models import Face, GeoPoint, Identity, MediaFile, MediaType


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


def test_get_media_paths_for_cluster_returns_jpegs_only(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media_a = _media("/photos/a.jpg")
    media_b = _media("/photos/b.JPEG")
    media_c = _media("/photos/c.heic")
    repo.save_media(media_a)
    repo.save_media(media_b)
    repo.save_media(media_c)

    face_a = _face(media_a.id)
    face_b = _face(media_b.id)
    face_c = _face(media_c.id)
    face_a.cluster_label = 7
    face_b.cluster_label = 7
    face_c.cluster_label = 7
    repo.save_face(face_a)
    repo.save_face(face_b)
    repo.save_face(face_c)

    paths = repo.get_media_paths_for_cluster(7, limit=10)

    assert sorted(paths) == ["/photos/a.jpg", "/photos/b.JPEG"]


def test_get_media_paths_for_cluster_includes_heic_when_requested(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media_a = _media("/photos/a.jpg")
    media_b = _media("/photos/b.heic")
    repo.save_media(media_a)
    repo.save_media(media_b)

    face_a = _face(media_a.id)
    face_b = _face(media_b.id)
    face_a.cluster_label = 7
    face_b.cluster_label = 7
    repo.save_face(face_a)
    repo.save_face(face_b)

    paths = repo.get_media_paths_for_cluster(
        7, limit=10, include_heic=True
    )

    assert sorted(paths) == ["/photos/a.jpg", "/photos/b.heic"]


def test_save_and_load_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    identity = Identity(id=str(uuid4()), name="Alice")
    repo.save_identity(identity)

    by_id = repo.get_identity_by_id(identity.id)
    by_name = repo.get_identity_by_name("Alice")

    assert by_id is not None
    assert by_id.name == "Alice"
    assert by_name is not None
    assert by_name.id == identity.id


def test_update_face_identity_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _media("/photos/face.jpg")
    repo.save_media(media)
    face = _face(media.id)
    repo.save_face(face)

    repo.update_face_identity(
        face.id,
        identity_id="identity-1",
        identity_name="Bob",
    )

    reloaded = repo.get_face_by_id(face.id)
    assert reloaded is not None
    assert reloaded.identity_id == "identity-1"
    assert reloaded.identity_name == "Bob"


def test_get_faces_for_cluster_and_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media = _media("/photos/face.jpg")
    repo.save_media(media)
    face = _face(media.id)
    face.cluster_label = 3
    face.identity_id = "identity-1"
    repo.save_face(face)

    assert len(repo.get_faces_for_cluster(3)) == 1
    assert len(repo.get_faces_for_cluster(4)) == 0
    assert len(repo.get_faces_for_identity("identity-1")) == 1
    assert len(repo.get_faces_for_identity("identity-2")) == 0


def test_get_identity_summary_counts_distinct_photos(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "test.db"
    repo = SqliteMediaRepository(str(db_path))

    media_a = _media("/photos/a.jpg")
    media_b = _media("/photos/b.jpg")
    repo.save_media(media_a)
    repo.save_media(media_b)

    face_a1 = _face(media_a.id)
    face_a2 = _face(media_a.id)
    face_b = _face(media_b.id)
    for face in (face_a1, face_a2, face_b):
        face.identity_id = "identity-1"
        face.identity_name = "Alice"
        repo.save_face(face)

    summaries = repo.get_identity_summary(limit=10)

    assert len(summaries) == 1
    assert summaries[0]["identity_id"] == "identity-1"
    assert summaries[0]["identity_name"] == "Alice"
    assert summaries[0]["face_count"] == 3
    assert summaries[0]["photo_count"] == 2
