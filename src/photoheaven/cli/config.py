"""Shared CLI configuration state and helpers."""

from __future__ import annotations

from pathlib import Path

DB_FILENAME = "photoheaven.db"

state: dict[str, str | None] = {"library": None}


def resolve_db_path(db_path: str | None) -> str:
    """Return the database path to use.

    Priority:
    1. Explicit ``--db`` value.
    2. ``--library`` global option or ``PHOTOHEAVEN_LIBRARY`` env var.
    3. Default ``<cwd>/db/photoheaven.db``.
    """
    if db_path:
        return db_path
    library = state.get("library")
    if library:
        return str(Path(library).expanduser().resolve() / DB_FILENAME)
    return str(Path.cwd() / "db" / DB_FILENAME)


def resolve_library_root(db_path: str) -> str:
    """Return the library root folder for a given database path.

    - ``<root>/photoheaven.db`` → ``<root>``
    - ``<root>/db/photoheaven.db`` → ``<root>``
    """
    path = Path(db_path).expanduser().resolve()
    if path.name != DB_FILENAME:
        return str(path.parent)
    if path.parent.name == "db":
        return str(path.parent.parent)
    return str(path.parent)


def resolve_photo_root(paths: list[str]) -> str | None:
    """Return the most likely photo root folder from stored media paths.

    The root is chosen by scoring each ancestor directory as
    ``count * depth``. This favours deep directories that contain many files,
    which correctly identifies the library root even when subfolders exist or
    a few outliers live elsewhere.
    """
    if not paths:
        return None

    dir_counts: dict[str, int] = {}
    for path_str in paths:
        path = Path(path_str)
        for parent in path.parents:
            parent_str = str(parent)
            dir_counts[parent_str] = dir_counts.get(parent_str, 0) + 1

    best_dir: str | None = None
    best_score = -1
    best_depth = -1
    for parent_str, count in dir_counts.items():
        depth = len(Path(parent_str).parts)
        score = count * depth
        if (
            score > best_score
            or (score == best_score and depth > best_depth)
        ):
            best_score = score
            best_dir = parent_str
            best_depth = depth

    return best_dir
