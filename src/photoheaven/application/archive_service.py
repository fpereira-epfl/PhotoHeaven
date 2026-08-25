"""Application service for archiving the library duplicates tree."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from photoheaven.adapters.integrity.hasher import Blake3Hasher
from photoheaven.application.ports import MediaRepository

logger = logging.getLogger(__name__)


@dataclass
class ArchiveResult:
    """Result of an archive run."""

    files_archived: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    files_removed: int = 0
    dirs_removed: int = 0
    bytes_transferred: int = 0
    failed_files: list[str] = field(default_factory=list)


@dataclass
class ArchiveProgress:
    """Live progress snapshot emitted during archiving."""

    stage: str = "scanning"
    total_files: int = 0
    files_done: int = 0
    current_file: str = ""
    bytes_total: int = 0
    bytes_done: int = 0
    errors: int = 0


class ArchiveService:
    """Move the library duplicates tree to an external archive path.

    The operation is resumable and tolerates flaky network/storage:

    * Before deleting a source file, the destination checksum is verified
      against the source checksum.
    * Copies are written to a ``.pharchive`` sibling file and renamed into
      place only after verification, so interrupted copies never leave a
      partial file at the final path.
    * Existing destination files with matching checksums are treated as
      already archived; the source file is removed.
    * Existing destination files with mismatched checksums are archived under
      a unique name instead of being overwritten.
    """

    def __init__(
        self,
        repository: MediaRepository,
        hasher: Blake3Hasher,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self.repository = repository
        self.hasher = hasher
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def archive_duplicates(
        self,
        source_root: Path,
        archive_root: Path,
        *,
        dry_run: bool = False,
        progress_callback: Optional[Callable[[ArchiveProgress], None]] = None,
    ) -> ArchiveResult:
        """Move all files from *source_root* into *archive_root*."""
        source_root = source_root.resolve()
        archive_root = archive_root.resolve()

        if not source_root.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_root}")
        if source_root == archive_root:
            raise ValueError("Source and archive roots must be different")

        result = ArchiveResult()
        progress = ArchiveProgress()
        self._notify(progress_callback, progress)

        files = sorted(
            [p for p in source_root.rglob("*") if p.is_file()],
            key=lambda p: str(p),
        )
        progress.stage = "archiving"
        progress.total_files = len(files)
        progress.bytes_total = sum(p.stat().st_size for p in files)
        self._notify(progress_callback, progress)

        for source_path in files:
            progress.current_file = source_path.name
            self._notify(progress_callback, progress)

            relative = source_path.relative_to(source_root)
            target_path = archive_root / relative
            file_size = source_path.stat().st_size

            try:
                archived = self._archive_one(
                    source_path, target_path, dry_run=dry_run
                )
            except Exception as exc:
                logger.error("Failed to archive %s: %s", source_path, exc)
                result.files_failed += 1
                result.failed_files.append(str(source_path))
                progress.errors += 1
                progress.files_done += 1
                self._notify(progress_callback, progress)
                continue

            if archived:
                result.files_archived += 1
                result.bytes_transferred += file_size
                progress.bytes_done += file_size
            else:
                result.files_skipped += 1

            progress.files_done += 1
            self._notify(progress_callback, progress)

        if not dry_run:
            progress.stage = "cleaning"
            progress.current_file = ""
            self._notify(progress_callback, progress)
            removed_dirs = self._remove_empty_dirs(source_root)
            result.dirs_removed = removed_dirs

        return result

    def _archive_one(
        self, source_path: Path, target_path: Path, *, dry_run: bool
    ) -> bool:
        """Archive a single file. Returns True if data was copied, False if skipped."""
        source_checksum = self._source_checksum(source_path)
        temp_path = target_path.with_name(target_path.name + ".pharchive")

        # Already archived to the exact target path.
        if target_path.exists():
            if self._same_file(source_path, target_path):
                if not dry_run:
                    self._remove_source(source_path)
                return False

            target_checksum = self._hash_file_with_retries(target_path)
            if target_checksum == source_checksum:
                if not dry_run:
                    self._remove_source(source_path)
                return False

            # Different content at the target path: archive under a unique name
            # rather than overwriting existing data.
            target_path = self._unique_target_path(target_path)
            temp_path = target_path.with_name(target_path.name + ".pharchive")

        # A previous run may have left a temp file behind. Reuse it if it
        # verifies, otherwise overwrite it.
        if temp_path.exists():
            temp_checksum = self._hash_file_with_retries(temp_path)
            if temp_checksum == source_checksum:
                if not dry_run:
                    self._replace_file(temp_path, target_path)
                    self._remove_source(source_path)
                return False
            # Partial/corrupt temp file: remove it and copy again.
            if not dry_run:
                temp_path.unlink()

        if dry_run:
            return True

        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._copy_file_with_retries(source_path, temp_path)
        temp_checksum = self._hash_file_with_retries(temp_path)
        if temp_checksum != source_checksum:
            raise RuntimeError(
                f"Checksum mismatch after copying {source_path} to {temp_path}"
            )
        self._replace_file(temp_path, target_path)
        self._remove_source(source_path)
        return True

    def _source_checksum(self, path: Path) -> str:
        """Return the checksum for *path*, preferring the DB record."""
        media = self.repository.get_by_path(str(path))
        if media is not None and media.checksum:
            return media.checksum
        return self._hash_file_with_retries(path)

    def _hash_file_with_retries(self, path: Path) -> str:
        """Compute a file checksum with transient-error retries."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self.hasher.hash_file(path)
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "Checksum attempt %d failed for %s: %s", attempt + 1, path, exc
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
        raise RuntimeError(f"Could not checksum {path}: {last_exc}") from last_exc

    def _copy_file_with_retries(self, source: Path, destination: Path) -> None:
        """Copy *source* to *destination*, retrying on transient errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self._copy_file(source, destination)
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Copy attempt %d failed for %s: %s", attempt + 1, source, exc
                )
                if destination.exists():
                    try:
                        destination.unlink()
                    except OSError:
                        pass
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
        raise RuntimeError(f"Could not copy {source} to {destination}: {last_exc}") from last_exc

    def _copy_file(self, source: Path, destination: Path) -> None:
        """Stream-copy a file with metadata preservation."""
        with open(source, "rb") as src, open(destination, "wb") as dst:
            shutil.copyfileobj(src, dst)
        shutil.copystat(source, destination)

    def _replace_file(self, source: Path, target: Path) -> None:
        """Atomically move *source* to *target*, even across filesystems."""
        shutil.move(str(source), str(target))

    def _remove_source(self, source_path: Path) -> None:
        """Delete the source file and its DB record."""
        try:
            if source_path.exists():
                source_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove source file %s: %s", source_path, exc)
        media = self.repository.get_by_path(str(source_path))
        if media is not None:
            self.repository.delete_media(media.id)

    def _same_file(self, a: Path, b: Path) -> bool:
        """Return True if *a* and *b* point to the same filesystem object."""
        try:
            return a.resolve() == b.resolve()
        except OSError:
            return False

    def _unique_target_path(self, target_path: Path) -> Path:
        """Return a nearby path that does not yet exist."""
        directory = target_path.parent
        stem = target_path.stem
        suffix = target_path.suffix
        n = 1
        while True:
            candidate = directory / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    def _remove_empty_dirs(self, root: Path) -> int:
        """Remove empty directories under *root*; return count removed."""
        removed = 0
        for dirpath, _dirnames, _filenames in os.walk(str(root), topdown=False):
            path = Path(dirpath)
            if path == root:
                continue
            try:
                if not any(path.iterdir()):
                    path.rmdir()
                    removed += 1
            except OSError:
                pass
        return removed

    def _notify(
        self,
        callback: Optional[Callable[[ArchiveProgress], None]],
        progress: ArchiveProgress,
    ) -> None:
        if callback is not None:
            callback(progress)
