"""Tests for library initialisation and migration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from photoheaven.adapters.integrity.hasher import Blake3Hasher
from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.library_service import LibraryMigrationService
from photoheaven.domain.models import MediaFile, MediaType


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _media(path: str) -> MediaFile:
    return MediaFile(
        id=str(uuid4()),
        path=path,
        checksum="abc",
        size_bytes=1,
        mtime=1.0,
        media_type=MediaType.IMAGE,
        capture_datetime=datetime.utcnow(),
    )


def test_init_library_creates_database(tmp_path: Path) -> None:
    service = LibraryMigrationService(Blake3Hasher())
    library = tmp_path / "MyLibrary.photoslibrary"

    db_path = service.init_library(library)

    assert db_path == library / "photoheaven.db"
    assert db_path.exists()
    # Opening the repository should succeed and see no tables issues.
    repo = SqliteMediaRepository(str(db_path))
    assert repo.count_media() == 0


def test_migrate_copies_files_and_updates_paths(tmp_path: Path) -> None:
    source = tmp_path / "Queue"
    library = tmp_path / "PhotoHeaven.photoslibrary"
    source_db = tmp_path / "db" / "photoheaven.db"

    photo = source / "2008" / "12" / "img.jpg"
    _write_file(photo, b"photo content")

    source_db.parent.mkdir(parents=True, exist_ok=True)
    repo = SqliteMediaRepository(str(source_db))
    repo.save_media(_media(str(photo)))

    service = LibraryMigrationService(Blake3Hasher())
    result = service.migrate(source, library, source_db)

    assert result.target_db_path == library / "photoheaven.db"
    assert result.files_copied == 1
    assert result.files_moved == 0
    assert result.files_skipped == 0
    assert result.errors == 0
    assert result.media_paths_updated == 1

    assert (library / "2008" / "12" / "img.jpg").exists()
    assert photo.exists()  # source left intact when not using --move

    target_repo = SqliteMediaRepository(str(result.target_db_path))
    media = target_repo.list_media(limit=10)[0]
    assert media.path == str(library / "2008" / "12" / "img.jpg")


def test_migrate_skips_duplicate_files(tmp_path: Path) -> None:
    source = tmp_path / "Queue"
    library = tmp_path / "PhotoHeaven.photoslibrary"
    source_db = tmp_path / "photoheaven.db"

    photo = source / "img.jpg"
    _write_file(photo, b"photo content")
    # Same file already in library.
    _write_file(library / "img.jpg", b"photo content")

    service = LibraryMigrationService(Blake3Hasher())
    result = service.migrate(source, library, source_db)

    assert result.files_copied == 0
    assert result.files_skipped == 1
    assert photo.exists()


def test_migrate_renames_on_collision(tmp_path: Path) -> None:
    source = tmp_path / "Queue"
    library = tmp_path / "PhotoHeaven.photoslibrary"
    source_db = tmp_path / "photoheaven.db"

    photo = source / "img.jpg"
    _write_file(photo, b"new content")
    _write_file(library / "img.jpg", b"old content")

    service = LibraryMigrationService(Blake3Hasher())
    result = service.migrate(source, library, source_db)

    assert result.files_copied == 1
    assert result.files_skipped == 0
    assert (library / "img_1.jpg").exists()


def test_migrate_moves_and_removes_source(tmp_path: Path) -> None:
    source = tmp_path / "Queue"
    library = tmp_path / "PhotoHeaven.photoslibrary"
    source_db = tmp_path / "db" / "photoheaven.db"

    photo = source / "img.jpg"
    _write_file(photo, b"photo content")

    source_db.parent.mkdir(parents=True, exist_ok=True)
    repo = SqliteMediaRepository(str(source_db))
    repo.save_media(_media(str(photo)))

    service = LibraryMigrationService(Blake3Hasher())
    result = service.migrate(source, library, source_db, move_files=True)

    assert result.files_moved == 1
    assert result.files_copied == 0
    assert not photo.exists()
    assert not source_db.exists()
    assert (library / "img.jpg").exists()


def test_migrate_dry_run_makes_no_changes(tmp_path: Path) -> None:
    source = tmp_path / "Queue"
    library = tmp_path / "PhotoHeaven.photoslibrary"

    photo = source / "img.jpg"
    _write_file(photo, b"photo content")

    service = LibraryMigrationService(Blake3Hasher())
    result = service.migrate(source, library, dry_run=True)

    assert result.files_copied == 1
    assert not (library / "img.jpg").exists()


def test_migrate_paths_only_updates_references_without_touching_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Queue"
    library = tmp_path / "PhotoHeaven.photoslibrary"
    source_db = tmp_path / "db" / "photoheaven.db"

    # Simulate a user who already moved files manually.
    photo_at_library = library / "img.jpg"
    _write_file(photo_at_library, b"photo content")

    source_db.parent.mkdir(parents=True, exist_ok=True)
    repo = SqliteMediaRepository(str(source_db))
    repo.save_media(_media(str(source / "img.jpg")))

    # Source folder no longer exists because files were already moved.
    assert not source.exists()

    service = LibraryMigrationService(Blake3Hasher())
    result = service.migrate(
        source, library, source_db, paths_only=True
    )

    assert result.files_copied == 0
    assert result.files_moved == 0
    assert result.files_skipped == 0
    assert result.media_paths_updated == 1

    target_repo = SqliteMediaRepository(str(result.target_db_path))
    media = target_repo.list_media(limit=10)[0]
    assert media.path == str(library / "img.jpg")


def test_migrate_rejects_same_source_and_library(tmp_path: Path) -> None:
    service = LibraryMigrationService(Blake3Hasher())
    source = tmp_path / "Queue"
    source.mkdir()

    try:
        service.migrate(source, source)
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "different" in str(exc).lower()
