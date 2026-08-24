"""Tests for library-aware CLI commands."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image
from typer.testing import CliRunner

from photoheaven.adapters.integrity.hasher import Blake3Hasher
from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.cli import config as cli_config
from photoheaven.cli.main import app
from photoheaven.domain.models import Face, MediaFile, MediaType

runner = CliRunner()


def _make_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color="red").save(path, "JPEG")


def _media(path: str, **kwargs: object) -> MediaFile:
    defaults = {
        "id": str(uuid4()),
        "path": path,
        "checksum": "abc",
        "size_bytes": 1,
        "mtime": datetime(2024, 5, 1, 12, 30, 45).timestamp(),
        "media_type": MediaType.IMAGE,
        "capture_datetime": datetime(2024, 5, 1, 12, 30, 45),
    }
    defaults.update(kwargs)
    return MediaFile(**defaults)


def test_rename_organize_defaults_to_library_files_root(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    SqliteMediaRepository(str(db_path))

    old_file = library / "files" / "2008" / "12" / "old.jpg"
    old_file.parent.mkdir(parents=True)
    old_file.write_bytes(b"not an image")

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rename", "--move", "--dry-run"])

    assert result.exit_code == 0
    assert "Planned renames/moves" in result.output
    assert "$/2008/12/old.jpg" in result.output
    assert "renamed │     1" in result.output
    cli_config.state["library"] = None


def test_rename_updates_stored_path_after_moving_file(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    repo = SqliteMediaRepository(str(db_path))

    old_file = library / "files" / "old.jpg"
    old_file.parent.mkdir(parents=True)
    old_file.write_bytes(b"not an image")

    checksum = Blake3Hasher().hash_file(old_file)
    repo.save_media(_media(str(old_file)))
    # Fix checksum so the repository can find the file after rename.
    media = repo.get_by_checksum("abc")
    assert media is not None
    media.checksum = checksum
    repo.save_media(media)

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rename"])

    assert result.exit_code == 0
    renamed = list((library / "files").glob("*.jpg"))
    assert len(renamed) == 1

    updated = repo.get_by_checksum(checksum)
    assert updated is not None
    assert updated.path == str(renamed[0])
    cli_config.state["library"] = None


def test_rename_skips_files_already_named_correctly(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    SqliteMediaRepository(str(db_path))

    base_mtime = datetime(2024, 5, 1, 12, 30, 45).timestamp()
    names = [
        "2024-05-01_12h30m45s.jpg",
        "2024-05-01_12h30m45s_1.jpg",
        "2024-05-01_12h30m45s_2.jpg",
    ]
    for name in names:
        path = library / "files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not an image")
        os.utime(path, (base_mtime, base_mtime))

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rename", "--dry-run"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert "skipped │     3" in result.output
    assert "renamed │     0" in result.output


def test_clean_defaults_to_library_files_root(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    SqliteMediaRepository(str(db_path))

    empty_dir = library / "files" / "empty"
    empty_dir.mkdir(parents=True)

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["clean"])

    assert result.exit_code == 0
    assert not empty_dir.exists()
    cli_config.state["library"] = None


def test_init_uses_configured_library(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert (library / "db" / "photoheaven.db").exists()
    assert (library / "files").exists()
    cli_config.state["library"] = None


def test_inspect_input_analyses_single_file(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    SqliteMediaRepository(str(db_path))

    source = tmp_path / "Queue" / "old.jpg"
    _make_jpeg(source)

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["inspect", "--input", str(source)])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert "Media info" in result.output
    assert source.name in result.output


def test_import_copies_file_into_library(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    SqliteMediaRepository(str(db_path))

    source = tmp_path / "Queue" / "old.jpg"
    _make_jpeg(source)

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["import", str(source)])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert not source.exists()  # default import absorbs the source
    imported = list((library / "files").rglob("*.jpg"))
    assert len(imported) == 1


def test_import_move_flag_removes_source(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    SqliteMediaRepository(str(db_path))

    source = tmp_path / "Queue" / "old.jpg"
    _make_jpeg(source)

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["import", str(source), "--move"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert not source.exists()
    imported = list((library / "files").rglob("*.jpg"))
    assert len(imported) == 1


def test_rename_include_faces_orders_names_by_importance(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    repo = SqliteMediaRepository(str(db_path))

    old_file = library / "files" / "2024-05-01_12h30m45s.jpg"
    old_file.parent.mkdir(parents=True)
    old_file.write_bytes(b"not an image")

    checksum = Blake3Hasher().hash_file(old_file)
    media = _media(str(old_file))
    media.checksum = checksum
    repo.save_media(media)

    # Alice appears in 2 photos, Bob in 1.
    repo.save_face(
        Face(
            id="face-1",
            media_id=media.id,
            bbox=(0, 0, 1, 1),
            embedding=[0.1],
            identity_name="Alice",
        )
    )
    repo.save_face(
        Face(
            id="face-2",
            media_id=media.id,
            bbox=(0, 0, 1, 1),
            embedding=[0.2],
            identity_name="Bob",
        )
    )
    other_media = _media(str(library / "files" / "other.jpg"))
    other_media.checksum = "other-checksum"
    repo.save_media(other_media)
    repo.save_face(
        Face(
            id="face-3",
            media_id=other_media.id,
            bbox=(0, 0, 1, 1),
            embedding=[0.3],
            identity_name="Alice",
        )
    )

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rename", "--include-faces"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    renamed = list((library / "files").glob("*.jpg"))
    assert len(renamed) == 1
    assert "Alice" in renamed[0].name
    assert "Bob" in renamed[0].name
    # Alice should appear before Bob because she is in more photos.
    alice_pos = renamed[0].name.find("Alice")
    bob_pos = renamed[0].name.find("Bob")
    assert alice_pos < bob_pos


def test_dedupe_list_shows_groups(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    repo = SqliteMediaRepository(str(db_path))

    source = library / "files" / "source.jpg"
    dup = library / "files" / "dup.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (50, 50), color="orange").save(source)
    shutil.copy2(source, dup)

    dt = datetime(2024, 5, 1, 12, 0, 0)
    repo.save_media(
        _media(str(source), checksum="c1", size_bytes=5000, capture_datetime=dt)
    )
    repo.save_media(
        _media(str(dup), checksum="c2", size_bytes=1000, capture_datetime=dt)
    )

    cli_config.state["library"] = str(library)
    runner.invoke(app, ["dedupe"])
    result = runner.invoke(app, ["dedupe", "--list"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert "Duplicate groups: 1" in result.output
    assert "★ keep" in result.output
    assert source.name in result.output
    assert dup.name in result.output


def test_dedupe_quiet_suppresses_progress(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    repo = SqliteMediaRepository(str(db_path))

    source = library / "files" / "source.jpg"
    dup = library / "files" / "dup.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (50, 50), color="orange").save(source)
    shutil.copy2(source, dup)

    dt = datetime(2024, 5, 1, 12, 0, 0)
    repo.save_media(
        _media(str(source), checksum="c1", size_bytes=5000, capture_datetime=dt)
    )
    repo.save_media(
        _media(str(dup), checksum="c2", size_bytes=1000, capture_datetime=dt)
    )

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["dedupe", "--quiet"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert "Groups found" in result.output
    assert "Comparing candidates" not in result.output


def test_dedupe_list_only_faces_shows_groups_with_faces(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True)
    repo = SqliteMediaRepository(str(db_path))

    face_source = library / "files" / "face_source.jpg"
    face_dup = library / "files" / "face_dup.jpg"
    noface_source = library / "files" / "noface_source.jpg"
    noface_dup = library / "files" / "noface_dup.jpg"
    face_source.parent.mkdir(parents=True)

    Image.new("RGB", (50, 50), color="orange").save(face_source)
    shutil.copy2(face_source, face_dup)
    Image.new("RGB", (50, 50), color="blue").save(noface_source)
    shutil.copy2(noface_source, noface_dup)

    dt = datetime(2024, 5, 1, 12, 0, 0)
    repo.save_media(
        _media(str(face_source), checksum="c1", size_bytes=5000, capture_datetime=dt)
    )
    repo.save_media(
        _media(str(face_dup), checksum="c2", size_bytes=1000, capture_datetime=dt)
    )
    repo.save_media(
        _media(str(noface_source), checksum="c3", size_bytes=5000, capture_datetime=dt)
    )
    repo.save_media(
        _media(str(noface_dup), checksum="c4", size_bytes=1000, capture_datetime=dt)
    )

    # Add a face only to one member of the face group.
    repo.save_face(
        Face(
            id="face-1",
            media_id=repo.get_by_checksum("c1").id,
            bbox=(0, 0, 10, 10),
            embedding=[0.1, 0.2, 0.3],
            detection_confidence=0.9,
        )
    )

    cli_config.state["library"] = str(library)
    runner.invoke(app, ["dedupe"])
    result = runner.invoke(app, ["dedupe", "--list", "--only-faces"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert "Duplicate groups: 1" in result.output
    assert "face_source.jpg" in result.output
    assert "noface_source.jpg" not in result.output
