"""Tests for CLI configuration helpers."""

from __future__ import annotations

from pathlib import Path

from photoheaven.cli import config as cli_config


def test_resolve_db_path_prefers_explicit_value() -> None:
    cli_config.state["library"] = None
    assert cli_config.resolve_db_path("/custom/db.db") == "/custom/db.db"


def test_resolve_db_path_uses_library_option(tmp_path: Path) -> None:
    cli_config.state["library"] = str(tmp_path / "MyLibrary.photoslibrary")
    assert cli_config.resolve_db_path(None) == str(
        tmp_path / "MyLibrary.photoslibrary" / "db" / "photoheaven.db"
    )
    cli_config.state["library"] = None


def test_resolve_library_package_for_library_db(tmp_path: Path) -> None:
    library = tmp_path / "MyLibrary.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    assert cli_config.resolve_library_package(str(db_path)) == str(library)


def test_resolve_library_package_falls_back_to_parent(tmp_path: Path) -> None:
    custom = tmp_path / "custom.db"
    assert cli_config.resolve_library_package(str(custom)) == str(tmp_path)


def test_resolve_photo_root_finds_common_folder() -> None:
    paths = [
        "/Users/francisco/Pictures/Queue/2008/12/img.jpg",
        "/Users/francisco/Pictures/Queue/2024/05/img.jpg",
        "/Users/francisco/Pictures/Queue/2024/06/img.jpg",
    ]
    assert cli_config.resolve_photo_root(paths) == "/Users/francisco/Pictures/Queue"


def test_resolve_photo_root_ignores_outliers() -> None:
    paths = [
        "/Users/francisco/Pictures/Queue/2008/12/img.jpg",
        "/Users/francisco/Pictures/Queue/2024/05/img.jpg",
        "/tmp/stray.jpg",
    ]
    assert cli_config.resolve_photo_root(paths) == "/Users/francisco/Pictures/Queue"


def test_resolve_photo_root_returns_none_for_empty_list() -> None:
    assert cli_config.resolve_photo_root([]) is None
