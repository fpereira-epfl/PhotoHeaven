"""Typer CLI entry point for PhotoHeaven."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from photoheaven.adapters.integrity.hasher import Blake3Hasher
from photoheaven.adapters.metadata.exif import FallbackMetadataExtractor
from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.ingestion_service import IngestionService, guess_media_type
from photoheaven.cli.faces import faces_app
from photoheaven.domain.models import MediaFile

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="photoheaven",
    help="Organise, analyse, and face-cluster local photos and videos.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)
app.add_typer(faces_app, name="faces")
console = Console()

# Make library log messages visible through the CLI.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)

SUPPORTED_EXTENSIONS = {
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".webp",
    ".avif",
    ".dng",
    # Videos
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".3gp",
    ".webm",
    ".mts",
    ".m2ts",
    ".ts",
    ".mpg",
    ".mpeg",
}


def _get_db_path(db_path: Optional[str]) -> str:
    if db_path:
        resolved = db_path
    else:
        resolved = str(Path.cwd() / "db" / "photoheaven.db")
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_service(db_path: str) -> IngestionService:
    repository = SqliteMediaRepository(db_path)
    return IngestionService(
        hasher=Blake3Hasher(),
        metadata_extractor=FallbackMetadataExtractor(),
        repository=repository,
    )


def _collect_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]

    if recursive:
        return [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    return [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _target_path_for_corrupted(
    file_path: Path, ingest_root: Path, target_root: Path
) -> Path:
    """Return a target path for a corrupted file, preserving relative structure."""
    try:
        relative = file_path.resolve().relative_to(ingest_root.resolve())
    except ValueError:
        relative = Path(file_path.name)

    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    # Handle name collisions by appending _1, _2, etc.
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    n = 1
    while True:
        candidate = target.parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest.", exists=True),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Ingest directories recursively."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-analyse files even if unchanged."
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
    move_corrupted_to: str | None = typer.Option(
        None,
        "--move-corrupted-to",
        help="Move files whose metadata could not be extracted to this directory.",
    ),
) -> None:
    """Ingest photos and videos into the library."""
    service = _build_service(_get_db_path(db_path))
    files = _collect_files(path, recursive)

    if not files:
        console.print("[yellow]No supported media files found.[/yellow]")
        raise typer.Exit(0)

    corrupted_target: Path | None = None
    if move_corrupted_to:
        corrupted_target = Path(move_corrupted_to).expanduser().resolve()
        corrupted_target.mkdir(parents=True, exist_ok=True)

    counts = {"added": 0, "updated": 0, "skipped": 0, "error": 0, "moved": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Ingesting {len(files)} item(s)...", total=len(files))
        for file_path in files:
            result = service.ingest_file(file_path, force=force)
            counts[result.status] = counts.get(result.status, 0) + 1

            should_move = False
            if corrupted_target is not None:
                if not result.metadata_extracted and result.status in {
                    "added",
                    "updated",
                }:
                    should_move = True
                elif result.status == "skipped":
                    # Already-ingested files may have been created before we
                    # tracked metadata_extracted. Re-check them so corrupted
                    # files can be quarantined on a subsequent run.
                    if not service.check_metadata(file_path):
                        should_move = True
                        if result.media is not None:
                            result.media.metadata_extracted = False
                            try:
                                service.repository.save_media(result.media)
                            except Exception:
                                logger.exception(
                                    "Could not update metadata_extracted flag for %s",
                                    file_path,
                                )

            if should_move and corrupted_target is not None:
                try:
                    target = _target_path_for_corrupted(
                        file_path, path, corrupted_target
                    )
                    file_path.rename(target)
                    counts["moved"] += 1
                except OSError as exc:
                    logger.warning(
                        "Could not move corrupted file %s: %s", file_path, exc
                    )

            progress.advance(task)

    table = Table(title="Ingestion summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    for status, count in counts.items():
        table.add_row(status, str(count))
    console.print(table)


@app.command()
def info(
    db_path: Optional[str] = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
) -> None:
    """Show library statistics."""
    repository = SqliteMediaRepository(_get_db_path(db_path))
    media_count = repository.count_media()
    face_count = repository.count_faces()

    table = Table(title="Library overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Media files", str(media_count))
    table.add_row("Detected faces", str(face_count))
    table.add_row("Database", _get_db_path(db_path))
    console.print(table)


def _format_datetime(value: Optional[datetime]) -> str:
    if value is None:
        return "[dim]not available[/dim]"
    return value.isoformat(sep=" ", timespec="seconds")


def _format_size(size_bytes: int) -> str:
    """Return a human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _load_media_for_show(
    path: Path,
    repository: SqliteMediaRepository,
    hasher: Blake3Hasher,
    extractor: FallbackMetadataExtractor,
) -> tuple[MediaFile, str]:
    """Return media metadata plus a source label ('library' or 'file')."""
    checksum = hasher.hash_file(path)
    media = repository.get_by_checksum(checksum)
    if media is not None:
        return media, "library"

    media_type = guess_media_type(path)
    metadata = extractor.extract(path, media_type)
    stat = path.stat()
    media = MediaFile(
        id="not-ingested",
        path=str(path.resolve()),
        checksum=checksum,
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
        media_type=metadata.media_type,
        capture_datetime=metadata.capture_datetime,
        make=metadata.make,
        model=metadata.model,
        gps=metadata.gps,
    )
    return media, "file"


def _show_single(path: Path, media: MediaFile, source: str) -> None:
    table = Table(title=f"Media info — {path.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Source", source)
    table.add_row("Path", media.path)
    table.add_row("Checksum", f"{media.checksum[:16]}…")
    table.add_row("Media type", media.media_type.value)
    table.add_row("Size", _format_size(media.size_bytes))
    table.add_row("Modified", _format_datetime(datetime.fromtimestamp(media.mtime)))
    table.add_row("Capture date", _format_datetime(media.capture_datetime))
    table.add_row("Camera make", media.make or "[dim]not available[/dim]")
    table.add_row("Camera model", media.model or "[dim]not available[/dim]")

    if media.gps:
        table.add_row("GPS", f"{media.gps.latitude:.6f}, {media.gps.longitude:.6f}")
    else:
        table.add_row("GPS", "[dim]not available[/dim]")

    console.print(table)


def _show_folder(
    root: Path, files: list[Path], repository: SqliteMediaRepository
) -> None:
    hasher = Blake3Hasher()
    extractor = FallbackMetadataExtractor()

    table = Table(title=f"Media overview — {root}")
    table.add_column("Path", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Size", justify="right", style="green", no_wrap=True)
    table.add_column("Capture date", style="yellow")
    table.add_column("Camera", style="blue")
    table.add_column("GPS", style="red")
    table.add_column("Source", style="white")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Reading {len(files)} item(s)...", total=len(files))
        for file_path in sorted(files):
            try:
                media, source = _load_media_for_show(file_path, repository, hasher, extractor)
            except Exception as exc:
                progress.advance(task)
                continue

            display_path = str(file_path.relative_to(root))
            camera = f"{media.make or ''} {media.model or ''}".strip() or "—"
            gps = f"{media.gps.latitude:.4f},{media.gps.longitude:.4f}" if media.gps else "—"
            table.add_row(
                display_path,
                media.media_type.value,
                _format_size(media.size_bytes),
                _format_datetime(media.capture_datetime),
                camera,
                gps,
                source,
            )
            progress.advance(task)

    console.print(table)


@app.command()
def show(
    path: Path = typer.Argument(
        ..., help="Photo, video, or folder to inspect.", exists=True
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Inspect directories recursively."
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
) -> None:
    """Show metadata for a photo/video or a folder of them."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)

    if path.is_file():
        media, source = _load_media_for_show(
            path, repository, Blake3Hasher(), FallbackMetadataExtractor()
        )
        _show_single(path, media, source)
        return

    files = _collect_files(path, recursive)
    if not files:
        console.print("[yellow]No supported media files found.[/yellow]")
        raise typer.Exit(0)

    _show_folder(path, files, repository)


def _capture_datetime_for_rename(path: Path) -> datetime:
    """Return capture date if available, otherwise filesystem modified time."""
    media_type = guess_media_type(path)
    metadata = FallbackMetadataExtractor().extract(path, media_type)
    if metadata.capture_datetime is not None:
        return metadata.capture_datetime
    return datetime.fromtimestamp(path.stat().st_mtime)


def _unique_target_name(
    base: str, ext: str, directory: Path, used_names: dict[Path, set[str]]
) -> str:
    """Return a unique file name in *directory*, adding _1, _2, etc. if needed."""
    used = used_names.setdefault(directory, set())

    def _available(name: str) -> bool:
        return name not in used and not (directory / name).exists()

    target_name = f"{base}{ext}"
    if _available(target_name):
        used.add(target_name)
        return target_name

    n = 1
    while True:
        candidate = f"{base}_{n}{ext}"
        if _available(candidate):
            used.add(candidate)
            return candidate
        n += 1


@app.command()
def rename(
    path: Path = typer.Argument(
        ..., help="Photo, video, or folder to rename.", exists=True
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Rename files in subfolders too."
    ),
    organize: Optional[str] = typer.Option(
        None,
        "--organize",
        "-o",
        help="Move files into YYYY/MM folders under this root path.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show planned renames/moves without actually doing them.",
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
) -> None:
    """Rename photos/videos to YYYY-MM-DD_HHhMMmSSs.<ext> based on capture date."""
    _ = _get_db_path(db_path)  # ensures default db dir exists, even if unused here

    files = _collect_files(path, recursive)
    if not files:
        console.print("[yellow]No supported media files found.[/yellow]")
        raise typer.Exit(0)

    if organize:
        root = Path(organize).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = path if path.is_dir() else path.parent

    used_names: dict[Path, set[str]] = {}
    plans: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    planning_errors: list[tuple[Path, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Planning {len(files)} rename(s)...", total=len(files))
        for file_path in sorted(files):
            try:
                dt = _capture_datetime_for_rename(file_path)
                base = dt.strftime("%Y-%m-%d_%Hh%Mm%Ss")
                ext = file_path.suffix.lower()

                # Already named with the correct date prefix — leave it alone.
                if file_path.stem.startswith(base) and not organize:
                    skipped.append(file_path)
                    progress.advance(task)
                    continue

                if organize:
                    target_dir = root / f"{dt:%Y}" / f"{dt:%m}"
                    target_dir.mkdir(parents=True, exist_ok=True)
                else:
                    target_dir = file_path.parent

                target_name = _unique_target_name(base, ext, target_dir, used_names)
                target = target_dir / target_name
                if target == file_path:
                    skipped.append(file_path)
                    progress.advance(task)
                    continue
                plans.append((file_path, target))
            except Exception as exc:
                planning_errors.append((file_path, str(exc)))
                logger.warning("Could not plan rename for %s: %s", file_path, exc)
            progress.advance(task)

    renamed_count = 0
    execution_errors: list[tuple[Path, str]] = []

    if dry_run:
        if plans:
            table = Table(title="Planned renames/moves")
            table.add_column("Current name", style="cyan", no_wrap=True)
            table.add_column("New name", style="magenta", no_wrap=True)
            for current, new in plans:
                table.add_row(str(current), str(new))
            console.print(table)
        renamed_count = len(plans)
    else:
        for current, new in plans:
            try:
                current.rename(new)
                console.print(f"✅ {current} → {new}")
                renamed_count += 1
            except Exception as exc:
                execution_errors.append((current, str(exc)))
                logger.warning("Rename failed for %s: %s", current, exc)
        for file_path in skipped:
            console.print(f"⏭️  {file_path} (already named)")
        for file_path, exc in planning_errors:
            console.print(f"❌ {file_path}: {exc}")
        for file_path, exc in execution_errors:
            console.print(f"❌ {file_path}: {exc}")

    counts = {
        "renamed": renamed_count,
        "skipped": len(skipped),
        "error": len(planning_errors) + len(execution_errors),
    }

    table = Table(title="Rename summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    for status, count in counts.items():
        table.add_row(status, str(count))
    console.print(table)


_METADATA_FILES = {".DS_Store", "Thumbs.db"}


def _is_effectively_empty(directory: Path) -> tuple[bool, list[Path]]:
    """Return (is_empty, metadata_files_to_remove).

    A directory is considered empty if it contains no real files and its only
    remaining contents are known metadata files such as .DS_Store or Thumbs.db.
    """
    entries = list(directory.iterdir())
    ignored = [p for p in entries if p.is_file() and p.name in _METADATA_FILES]
    remaining = [p for p in entries if p not in ignored]
    return len(remaining) == 0, ignored


@app.command()
def clean(
    path: Path = typer.Argument(
        ..., help="Root folder to scan for empty directories.", exists=True, file_okay=False
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively remove empty subdirectories.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="List empty directories without removing them.",
    ),
) -> None:
    """Remove empty folders under a directory (ignoring .DS_Store / Thumbs.db)."""
    if not path.is_dir():
        console.print("[red]Path must be a directory.[/red]")
        raise typer.Exit(1)

    root = path.resolve()
    removed: list[Path] = []
    errors: list[tuple[Path, str]] = []

    def _remove_directory(directory: Path, ignored: list[Path]) -> None:
        try:
            for metadata_file in ignored:
                metadata_file.unlink()
            directory.rmdir()
            removed.append(directory)
            suffix = f" (ignored {', '.join(p.name for p in ignored)})" if ignored else ""
            console.print(f"🗑️  Removed {directory}{suffix}")
        except OSError as exc:
            errors.append((directory, str(exc)))

    if recursive:
        if dry_run:
            # Simulate bottom-up removal: a directory is removed if it is
            # effectively empty and all of its subdirectories are also marked
            # for removal.
            to_remove: set[Path] = set()
            for parent_str, _dirs, _files in os.walk(str(root), topdown=False):
                parent = Path(parent_str).resolve()
                if parent == root:
                    continue
                entries = list(parent.iterdir())
                ignored = [p for p in entries if p.is_file() and p.name in _METADATA_FILES]
                remaining = [p for p in entries if p not in ignored]
                if not any(p for p in remaining if p not in to_remove):
                    to_remove.add(parent)
            removed = sorted(to_remove)
            console.print(
                f"[cyan]Dry run — {len(removed)} empty director(y/ies) would be removed:[/cyan]"
            )
            for directory in removed:
                console.print(f"🗑️  {directory}")
        else:
            # Walk bottom-up so children are removed before their parents.
            for parent_str, _dirs, _files in os.walk(str(root), topdown=False):
                parent = Path(parent_str).resolve()
                if parent == root:
                    continue
                empty, ignored = _is_effectively_empty(parent)
                if empty:
                    _remove_directory(parent, ignored)
    else:
        # Only check immediate subdirectories of the root.
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            empty, ignored = _is_effectively_empty(child)
            if not empty:
                continue
            if dry_run:
                removed.append(child)
                console.print(f"🗑️  {child}")
            else:
                _remove_directory(child, ignored)

    for directory, exc in errors:
        console.print(f"❌ {directory}: {exc}")

    table = Table(title="Clean summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("removed" if not dry_run else "would remove", str(len(removed)))
    table.add_row("error", str(len(errors)))
    console.print(table)


@app.command()
def version() -> None:
    """Print the PhotoHeaven version."""
    from photoheaven import __version__

    console.print(f"PhotoHeaven {__version__}")
