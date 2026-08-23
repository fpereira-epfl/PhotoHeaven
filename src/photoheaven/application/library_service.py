"""Application service for library initialisation and migration."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.ports import Hasher

logger = logging.getLogger(__name__)

DB_FILENAME = "photoheaven.db"


@dataclass(frozen=True)
class MigrationResult:
    """Result summary for a library migration."""

    target_db_path: Path
    files_copied: int
    files_moved: int
    files_skipped: int
    errors: int
    media_paths_updated: int


class LibraryMigrationService:
    """Create and migrate self-contained PhotoHeaven libraries.

    A library is a folder that contains both the media files and the
    ``photoheaven.db`` database. Migration preserves the relative folder
    structure of the source directory and updates stored media paths so the
    database remains valid after the move/copy.
    """

    def __init__(self, hasher: Hasher) -> None:
        self.hasher = hasher

    def init_library(self, library_path: Path) -> Path:
        """Create a new, empty library folder with a database.

        Returns the path to the created database.
        """
        library_path = library_path.expanduser().resolve()
        library_path.mkdir(parents=True, exist_ok=True)
        db_path = library_path / DB_FILENAME

        if db_path.exists():
            logger.info("Library database already exists at %s", db_path)
        else:
            # Creating the repository initialises the schema.
            SqliteMediaRepository(str(db_path))
            logger.info("Created library database at %s", db_path)

        return db_path

    def migrate(
        self,
        source_dir: Path,
        library_path: Path,
        source_db_path: Path | None = None,
        *,
        move_files: bool = False,
        paths_only: bool = False,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Copy/move media files and a database into a library folder.

        Args:
            source_dir: Root folder containing the media files (or the old
                root when ``paths_only`` is True).
            library_path: Target library folder.
            source_db_path: Path to the existing database. If omitted, defaults
                to ``<cwd>/db/photoheaven.db``.
            move_files: If True, physically move files to the library instead
                of copying them. Mutually exclusive with ``paths_only``.
            paths_only: If True, do not touch any media files. Only copy the
                database into the library and update stored media paths from
                ``source_dir`` to ``library_path``. Use this when you have
                already moved the files manually.
            dry_run: If True, report what would happen without making changes.

        Returns:
            A summary of migrated files and updated database paths.
        """
        if move_files and paths_only:
            raise ValueError(
                "--move-files and --paths-only are mutually exclusive"
            )

        source_dir = source_dir.expanduser().resolve()
        library_path = library_path.expanduser().resolve()
        library_path.mkdir(parents=True, exist_ok=True)
        target_db_path = library_path / DB_FILENAME

        if source_db_path is None:
            source_db_path = Path.cwd() / "db" / DB_FILENAME
        else:
            source_db_path = Path(source_db_path).expanduser().resolve()

        if source_dir == library_path:
            raise ValueError(
                "Source directory and library path must be different"
            )

        if not dry_run:
            self._prepare_target_database(
                source_db_path, target_db_path
            )

        files_copied = 0
        files_moved = 0
        files_skipped = 0
        errors = 0

        if not paths_only:
            for file_path in sorted(source_dir.rglob("*")):
                if not file_path.is_file():
                    continue

                try:
                    relative = file_path.relative_to(source_dir)
                except ValueError:
                    continue

                target_file = library_path / relative
                result = self._migrate_file(
                    file_path,
                    target_file,
                    move=move_files,
                    dry_run=dry_run,
                )
                if result == "copied":
                    files_copied += 1
                elif result == "moved":
                    files_moved += 1
                elif result == "skipped":
                    files_skipped += 1
                else:
                    errors += 1

        media_paths_updated = 0
        if not dry_run:
            repository = SqliteMediaRepository(str(target_db_path))
            media_paths_updated = self._update_media_paths(
                repository, source_dir, library_path
            )

        if move_files and not dry_run:
            self._remove_source_database(source_db_path, target_db_path)
            self._remove_empty_directories(source_dir)

        return MigrationResult(
            target_db_path=target_db_path,
            files_copied=files_copied,
            files_moved=files_moved,
            files_skipped=files_skipped,
            errors=errors,
            media_paths_updated=media_paths_updated,
        )

    def _prepare_target_database(
        self, source_db_path: Path, target_db_path: Path
    ) -> None:
        """Copy an existing database or create a fresh one in the library."""
        if source_db_path.exists():
            if (
                target_db_path.exists()
                and source_db_path.samefile(target_db_path)
            ):
                logger.info("Database already located in library")
                return
            shutil.copy2(source_db_path, target_db_path)
            logger.info("Copied database to %s", target_db_path)
        else:
            if not target_db_path.exists():
                SqliteMediaRepository(str(target_db_path))
                logger.info("Created new database at %s", target_db_path)

    def _migrate_file(
        self, source: Path, target: Path, *, move: bool, dry_run: bool
    ) -> str:
        """Migrate a single file and return its outcome label."""
        if dry_run:
            return "skipped" if target.exists() else (
                "moved" if move else "copied"
            )

        try:
            source_checksum = self.hasher.hash_file(source)
        except Exception:
            logger.exception("Could not hash source file %s", source)
            return "error"

        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            try:
                target_checksum = self.hasher.hash_file(target)
            except Exception:
                logger.exception("Could not hash existing target %s", target)
                return "error"

            if source_checksum == target_checksum:
                if move:
                    try:
                        source.unlink()
                    except OSError:
                        logger.exception(
                            "Could not remove source file %s", source
                        )
                return "skipped"

            target = self._unique_target(target)

        try:
            if move:
                shutil.move(str(source), str(target))
            else:
                shutil.copy2(source, target)
        except Exception:
            logger.exception("Could not migrate %s to %s", source, target)
            return "error"

        try:
            target_checksum = self.hasher.hash_file(target)
        except Exception:
            logger.exception("Could not verify target file %s", target)
            return "error"

        if source_checksum != target_checksum:
            logger.error(
                "Checksum mismatch after migrating %s to %s", source, target
            )
            return "error"

        return "moved" if move else "copied"

    def _unique_target(self, target: Path) -> Path:
        """Return a non-colliding target path by appending _1, _2, etc."""
        stem = target.stem
        suffix = target.suffix
        parent = target.parent
        n = 1
        while True:
            candidate = parent / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    def _update_media_paths(
        self,
        repository: SqliteMediaRepository,
        source_dir: Path,
        library_path: Path,
    ) -> int:
        """Update stored media paths from the source tree to the library."""
        updated = 0
        offset = 0
        limit = 1000

        while True:
            media_batch = repository.list_media(limit=limit, offset=offset)
            if not media_batch:
                break

            for media in media_batch:
                media_path = Path(media.path)
                try:
                    relative = media_path.relative_to(source_dir)
                except ValueError:
                    # Path is not under the source directory; leave it as-is.
                    continue

                new_path = str(library_path / relative)
                if new_path == media.path:
                    continue

                media.path = new_path
                media.updated_at = datetime.utcnow()
                repository.save_media(media)
                updated += 1

            if len(media_batch) < limit:
                break
            offset += limit

        logger.info("Updated %d media path(s) in database", updated)
        return updated

    def _remove_source_database(
        self, source_db_path: Path, target_db_path: Path
    ) -> None:
        """Delete the source database after a successful move."""
        if not source_db_path.exists():
            return
        if (
            target_db_path.exists()
            and source_db_path.samefile(target_db_path)
        ):
            return

        try:
            source_db_path.unlink()
            logger.info("Removed source database %s", source_db_path)
        except OSError:
            logger.exception(
                "Could not remove source database %s", source_db_path
            )

    def _remove_empty_directories(self, root: Path) -> None:
        """Remove empty directories left behind after moving files."""
        if not root.exists():
            return

        directories = [p for p in root.rglob("*") if p.is_dir()]
        directories.sort(key=lambda p: len(p.parts), reverse=True)

        for directory in directories:
            if directory == root:
                continue
            try:
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                logger.exception(
                    "Could not remove empty directory %s", directory
                )
