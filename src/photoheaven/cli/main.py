"""Typer CLI entry point for PhotoHeaven."""

from __future__ import annotations

import logging
import os
import re
import shutil
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
from photoheaven.application.library_service import LibraryService
from photoheaven.cli import config as cli_config
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


@app.callback()
def main(
    library: Optional[str] = typer.Option(
        None,
        "--library",
        envvar="PHOTOHEAVEN_LIBRARY",
        help="Path to a self-contained PhotoHeaven library folder.",
    ),
) -> None:
    """Global options for all PhotoHeaven commands."""
    if library:
        cli_config.state["library"] = library

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


def _get_db_path() -> str:
    if not cli_config.state.get("library"):
        console.print(
            "[red]PHOTOHEAVEN_LIBRARY is not set. Use "
            "`export PHOTOHEAVEN_LIBRARY=<path>` or pass "
            "`--library <path>` before the command.[/red]"
        )
        raise typer.Exit(1)
    resolved = cli_config.resolve_db_path(None)
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_target_path(
    path: Optional[Path], *, command_name: str
) -> Path:
    """Return the filesystem target for a command.

    Uses the explicit ``path`` if given, otherwise falls back to
    ``<library>/files`` when ``PHOTOHEAVEN_LIBRARY`` / ``--library`` is set.
    """
    if path is not None:
        return path.expanduser().resolve()

    library_files = cli_config.resolve_library_files_root()
    if library_files is None:
        console.print(
            f"[red]{command_name} requires PHOTOHEAVEN_LIBRARY. Use "
            "`export PHOTOHEAVEN_LIBRARY=<path>` or pass "
            "`--library <path>` before the command.[/red]"
        )
        raise typer.Exit(1)

    return library_files


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
def sync(
    force: bool = typer.Option(
        False, "--force", help="Re-analyse files even if checksums match."
    ),
    prune: bool = typer.Option(
        False, "--prune", help="Remove DB records for files no longer on disk."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be synced without making changes."
    ),
) -> None:
    """Synchronise the database with the library files tree."""
    target_path = _resolve_target_path(None, command_name="sync")
    service = _build_service(_get_db_path())
    files = _collect_files(target_path, recursive=True)

    if not files:
        console.print("[yellow]No supported media files found.[/yellow]")
        raise typer.Exit(0)

    corrupted_target = target_path.parent / "corrupted"
    corrupted_target.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "error": 0,
        "moved": 0,
        "pruned": 0,
    }
    seen_paths: set[str] = set()

    if dry_run:
        hasher = Blake3Hasher()
        for file_path in files:
            seen_paths.add(str(file_path))
            try:
                checksum = hasher.hash_file(file_path)
                stat = file_path.stat()
                existing = service.repository.get_by_checksum(checksum)
                if existing is None:
                    counts["added"] += 1
                elif existing.mtime != stat.st_mtime or force:
                    counts["updated"] += 1
                else:
                    counts["skipped"] += 1
            except Exception:
                counts["error"] += 1

        if prune:
            for media_path in service.repository.get_all_media_paths():
                if str(media_path).startswith(str(target_path)) and not Path(
                    media_path
                ).exists():
                    counts["pruned"] += 1
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Syncing {len(files)} item(s)...", total=len(files)
            )
            for file_path in files:
                seen_paths.add(str(file_path))
                result = service.ingest_file(file_path, force=force)
                counts[result.status] = counts.get(result.status, 0) + 1

                should_move = False
                if not result.metadata_extracted and result.status in {
                    "added",
                    "updated",
                }:
                    should_move = True
                elif result.status == "skipped":
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

                if should_move:
                    try:
                        target = _target_path_for_corrupted(
                            file_path, target_path, corrupted_target
                        )
                        file_path.rename(target)
                        if result.media is not None:
                            result.media.path = str(target)
                            result.media.metadata_extracted = False
                            try:
                                service.repository.save_media(result.media)
                            except Exception:
                                logger.exception(
                                    "Could not update path for corrupted file %s",
                                    target,
                                )
                        counts["moved"] += 1
                    except OSError as exc:
                        logger.warning(
                            "Could not move corrupted file %s: %s", file_path, exc
                        )

                progress.advance(task)

        if prune:
            for media in service.repository.list_media(limit=1_000_000):
                media_path = Path(media.path)
                if (
                    str(media_path).startswith(str(target_path))
                    and not media_path.exists()
                ):
                    try:
                        service.repository.delete_media(media.id)
                        counts["pruned"] += 1
                    except Exception:
                        logger.exception(
                            "Could not prune missing media %s", media.id
                        )

    table = Table(title="Sync summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    for status in ("added", "updated", "skipped", "moved", "pruned", "error"):
        table.add_row(status, str(counts.get(status, 0)))
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no changes were made.[/yellow]")


@app.command(name="import")
def import_(
    source: Path = typer.Argument(
        ..., help="External file or folder to import.", exists=True
    ),
    move: bool = typer.Option(
        False,
        "--move",
        "-mv",
        help="Physically move files into the library instead of copying them.",
    ),
    recursive: bool = typer.Option(
        True, "--recursive", "-r", help="Scan subdirectories."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show planned imports without making changes."
    ),
) -> None:
    """Import photos/videos from an external folder into the library."""
    library_files_root = _resolve_target_path(None, command_name="import")
    library_root = library_files_root.parent
    service = _build_service(_get_db_path())
    hasher = Blake3Hasher()

    source = source.expanduser().resolve()
    files = (
        [source]
        if source.is_file()
        else _collect_files(source, recursive)
    )

    if not files:
        console.print("[yellow]No supported media files found.[/yellow]")
        raise typer.Exit(0)

    corrupted_target = library_root / "corrupted"
    corrupted_target.mkdir(parents=True, exist_ok=True)

    used_names: dict[Path, set[str]] = {}
    counts = {"imported": 0, "corrupted": 0, "error": 0}

    def _target_for(file_path: Path) -> Path:
        dt = _capture_datetime_for_rename(file_path)
        base = dt.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        ext = file_path.suffix.lower()
        target_dir = library_files_root / f"{dt:%Y}" / f"{dt:%m}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = _unique_target_name(base, ext, target_dir, used_names)
        return target_dir / target_name

    if dry_run:
        for file_path in files:
            target = _target_for(file_path)
            console.print(f"{file_path} → {target}")
            counts["imported"] += 1

        table = Table(title="Import summary")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        table.add_row("would import", str(counts["imported"]))
        console.print(table)
        console.print("[yellow]Dry run — no changes were made.[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"Importing {len(files)} item(s)...", total=len(files)
        )
        for file_path in files:
            try:
                source_checksum = hasher.hash_file(file_path)
                target = _target_for(file_path)

                if move:
                    shutil.move(str(file_path), str(target))
                else:
                    shutil.copy2(str(file_path), str(target))

                target_checksum = hasher.hash_file(target)
                if source_checksum != target_checksum:
                    logger.error(
                        "Checksum mismatch after importing %s to %s",
                        file_path,
                        target,
                    )
                    counts["error"] += 1
                    progress.advance(task)
                    continue

                if not move:
                    try:
                        file_path.unlink()
                    except OSError as exc:
                        logger.warning(
                            "Could not remove source file %s: %s", file_path, exc
                        )

                result = service.ingest_file(target, force=False)

                if not result.metadata_extracted:
                    corrupted_path = _target_path_for_corrupted(
                        target, library_files_root, corrupted_target
                    )
                    target.rename(corrupted_path)
                    counts["corrupted"] += 1
                    if result.media is not None:
                        result.media.path = str(corrupted_path)
                        result.media.metadata_extracted = False
                        try:
                            service.repository.save_media(result.media)
                        except Exception:
                            logger.exception(
                                "Could not update path for corrupted import %s",
                                corrupted_path,
                            )
                else:
                    counts["imported"] += 1
            except Exception:
                logger.exception("Could not import %s", file_path)
                counts["error"] += 1

            progress.advance(task)

    table = Table(title="Import summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("imported", str(counts["imported"]))
    table.add_row("corrupted", str(counts["corrupted"]))
    table.add_row("error", str(counts["error"]))
    console.print(table)


@app.command()
def info() -> None:
    """Show library statistics."""
    resolved_db = _get_db_path()
    repository = SqliteMediaRepository(resolved_db)
    media_count = repository.count_media()
    face_count = repository.count_faces()

    paths = repository.get_all_media_paths()
    photo_root = cli_config.resolve_photo_root(paths)
    package_root = cli_config.resolve_library_package(resolved_db)
    if photo_root is None:
        photo_root = str(Path(package_root) / cli_config.FILES_DIR)

    table = Table(title="Library overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Library package", package_root)
    table.add_row("Photo root", photo_root)
    table.add_row("Media files", str(media_count))
    table.add_row("Detected faces", str(face_count))
    table.add_row("Database", resolved_db)
    console.print(table)


_REBASE_DATE_RE = re.compile(r"^(.*)/\d{4}/\d{2}(?:/|$)")


def _rebase_path(path: str, new_root: str) -> str:
    """Return path rebased from its old root to ``new_root``.

    The old root is the absolute path prefix before the first ``YYYY/MM``
    segment. For example ``/Queue/2008/12/img.jpg`` becomes
    ``<new_root>/2008/12/img.jpg``.
    """
    match = _REBASE_DATE_RE.match(path)
    if not match:
        return path

    suffix_start = len(match.group(1)) + 1  # skip the matched '/'
    suffix = path[suffix_start:]
    return f"{new_root}/{suffix}"


@app.command()
def rebase(
    debug: bool = typer.Option(
        False, "--debug", help="Show why paths were left unchanged."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would happen without making changes."
    ),
) -> None:
    """Rebase all stored media paths to the library root."""
    new_root = str(_resolve_target_path(None, command_name="rebase"))
    resolved_db = _get_db_path()
    repository = SqliteMediaRepository(resolved_db)
    paths = repository.get_all_media_paths()

    if not paths:
        console.print("[yellow]No media paths in database.[/yellow]")
        raise typer.Exit(0)
    updated = 0
    unchanged = 0
    unchanged_reasons: dict[str, int] = {}
    unchanged_samples: dict[str, list[str]] = {}
    _DEBUG_SAMPLE_LIMIT = 20

    def _record_unchanged(media_path: str, reason: str) -> None:
        nonlocal unchanged
        unchanged += 1
        unchanged_reasons[reason] = unchanged_reasons.get(reason, 0) + 1
        if debug:
            samples = unchanged_samples.setdefault(reason, [])
            if len(samples) < _DEBUG_SAMPLE_LIMIT:
                samples.append(media_path)

    def _unchanged_reason(media_path: str) -> str:
        if media_path == new_root or media_path.startswith(new_root + "/"):
            return "already under new root"
        if _REBASE_DATE_RE.search(media_path) is None:
            return "no YYYY/MM folder segment found"
        return "other"

    if dry_run:
        for media_path in paths:
            new_path = _rebase_path(media_path, new_root)
            if new_path != media_path:
                updated += 1
            else:
                _record_unchanged(media_path, _unchanged_reason(media_path))
    else:
        offset = 0
        limit = 1000
        while True:
            batch = repository.list_media(limit=limit, offset=offset)
            if not batch:
                break

            for media in batch:
                new_path = _rebase_path(media.path, new_root)
                if new_path == media.path:
                    _record_unchanged(
                        media.path, _unchanged_reason(media.path)
                    )
                    continue

                media.path = new_path
                media.updated_at = datetime.utcnow()
                repository.save_media(media)
                updated += 1

            if len(batch) < limit:
                break
            offset += limit

    table = Table(title="Rebase summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("New root", new_root)
    table.add_row("Paths updated", str(updated))
    table.add_row("Paths unchanged", str(unchanged))
    console.print(table)

    if debug and unchanged_reasons:
        debug_table = Table(title="Unchanged path reasons")
        debug_table.add_column("Reason", style="cyan")
        debug_table.add_column("Count", justify="right", style="magenta")
        for reason, count in sorted(unchanged_reasons.items()):
            debug_table.add_row(reason, str(count))
        console.print(debug_table)

        for reason, samples in unchanged_samples.items():
            sample_table = Table(
                title=f"Sample unchanged paths — {reason} (first {len(samples)})"
            )
            sample_table.add_column("Path", style="cyan")
            for sample_path in samples:
                sample_table.add_row(sample_path)
            console.print(sample_table)

    if dry_run:
        console.print("[yellow]Dry run — no changes were made.[/yellow]")


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
            except Exception:
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


@app.command(name="inspect")
def inspect(
    input: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Inspect a single file instead of the whole library.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Inspect metadata for all photos/videos in the library, or one file."""
    db = _get_db_path()
    repository = SqliteMediaRepository(db)

    if input is not None:
        file_path = input.expanduser().resolve()
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            console.print("[red]Unsupported file type.[/red]")
            raise typer.Exit(1)

        hasher = Blake3Hasher()
        extractor = FallbackMetadataExtractor()
        try:
            media, source = _load_media_for_show(file_path, repository, hasher, extractor)
        except Exception as exc:
            logger.exception("Could not inspect %s", file_path)
            console.print(f"[red]Could not inspect file:[/red] {exc}")
            raise typer.Exit(1) from exc

        _show_single(file_path, media, source)
        return

    target_path = _resolve_target_path(None, command_name="inspect")
    files = _collect_files(target_path, recursive=True)
    if not files:
        console.print("[yellow]No supported media files found.[/yellow]")
        raise typer.Exit(0)

    _show_folder(target_path, files, repository)


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
    move: bool = typer.Option(
        False,
        "--move",
        "-mv",
        help="Move renamed files into YYYY/MM folders under the library files root.",
    ),
    include_faces: bool = typer.Option(
        False,
        "--include-faces",
        help="Append recognised face names to the filename, ordered by importance.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show planned renames/moves without making changes."
    ),
) -> None:
    """Rename photos/videos to YYYY-MM-DD_HHhMMmSSs.<ext> based on capture date."""
    resolved_db = _get_db_path()
    repository = SqliteMediaRepository(resolved_db)
    hasher = Blake3Hasher()

    source_path = _resolve_target_path(None, command_name="rename")
    files = _collect_files(source_path, recursive=True)
    if not files:
        console.print("[yellow]No supported media files found.[/yellow]")
        raise typer.Exit(0)

    root = source_path
    root.mkdir(parents=True, exist_ok=True)

    identity_importance = repository.get_identity_photo_counts()
    used_names: dict[Path, set[str]] = {}
    plans: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    planning_errors: list[tuple[Path, str]] = []

    def _display_path(path: Path) -> str:
        """Return a short display path rooted at the library files folder."""
        try:
            relative = path.resolve().relative_to(root.resolve())
            return f"$/{relative}"
        except ValueError:
            return str(path)

    def _face_names_for(path: Path, checksum: str) -> list[str]:
        if not include_faces:
            return []
        media = repository.get_by_checksum(checksum)
        if media is None:
            return []
        faces = repository.list_faces_for_media(media.id)
        names = {face.identity_name for face in faces if face.identity_name}
        return sorted(
            names,
            key=lambda name: (-identity_importance.get(name, 0), name),
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"Planning {len(files)} rename(s)...", total=len(files)
        )
        for file_path in sorted(files):
            try:
                dt = _capture_datetime_for_rename(file_path)
                base = dt.strftime("%Y-%m-%d_%Hh%Mm%Ss")
                ext = file_path.suffix.lower()

                checksum = hasher.hash_file(file_path)
                face_names = _face_names_for(file_path, checksum)
                if face_names:
                    base = f"{base}_{'_'.join(face_names)}"

                if move:
                    target_dir = root / f"{dt:%Y}" / f"{dt:%m}"
                    target_dir.mkdir(parents=True, exist_ok=True)
                else:
                    target_dir = file_path.parent

                ideal_target = target_dir / f"{base}{ext}"
                if ideal_target.resolve() == file_path.resolve():
                    skipped.append(file_path)
                    progress.advance(task)
                    continue

                target_name = _unique_target_name(
                    base, ext, target_dir, used_names
                )
                target = target_dir / target_name
                plans.append((file_path, target, checksum))
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
            for current, new, _ in plans:
                table.add_row(_display_path(current), _display_path(new))
            console.print(table)
        renamed_count = len(plans)
    else:
        for current, new, checksum in plans:
            try:
                current.rename(new)
                console.print(f"✅ {_display_path(current)} → {_display_path(new)}")
                renamed_count += 1

                media = repository.get_by_checksum(checksum)
                if media is not None:
                    media.path = str(new)
                    media.updated_at = datetime.utcnow()
                    try:
                        repository.save_media(media)
                    except Exception:
                        logger.exception(
                            "Could not update stored path for %s", new
                        )
            except Exception as exc:
                execution_errors.append((current, str(exc)))
                logger.warning("Rename failed for %s: %s", current, exc)
        for file_path in skipped:
            console.print(f"⏭️  {_display_path(file_path)} (already named)")
        for file_path, exc in planning_errors:
            console.print(f"❌ {_display_path(file_path)}: {exc}")
        for file_path, exc in execution_errors:
            console.print(f"❌ {_display_path(file_path)}: {exc}")

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

    if dry_run:
        console.print("[yellow]Dry run — no changes were made.[/yellow]")


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
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List empty directories without removing them."
    ),
) -> None:
    """Remove empty folders under the library files tree (ignoring .DS_Store / Thumbs.db)."""
    target_path = _resolve_target_path(None, command_name="clean")
    if not target_path.is_dir():
        console.print("[red]Path must be a directory.[/red]")
        raise typer.Exit(1)

    root = target_path.resolve()
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

    if dry_run:
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
        for parent_str, _dirs, _files in os.walk(str(root), topdown=False):
            parent = Path(parent_str).resolve()
            if parent == root:
                continue
            empty, ignored = _is_effectively_empty(parent)
            if empty:
                _remove_directory(parent, ignored)

    for directory, exc in errors:
        console.print(f"❌ {directory}: {exc}")

    table = Table(title="Clean summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("removed" if not dry_run else "would remove", str(len(removed)))
    table.add_row("error", str(len(errors)))
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no changes were made.[/yellow]")


@app.command()
def init() -> None:
    """Create a new self-contained PhotoHeaven library."""
    library = cli_config.state.get("library")
    if library is None:
        console.print(
            "[red]init requires PHOTOHEAVEN_LIBRARY. Use "
            "`export PHOTOHEAVEN_LIBRARY=<path>` or pass "
            "`--library <path>` before the command.[/red]"
        )
        raise typer.Exit(1)
    library_path = Path(library)

    service = LibraryService(Blake3Hasher())

    try:
        db_path = service.init_library(library_path)
    except Exception as exc:
        logger.exception("Failed to initialise library")
        console.print(f"[red]Failed to initialise library:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"Created library database at [cyan]{db_path}[/cyan]")


@app.command()
def version() -> None:
    """Print the PhotoHeaven version."""
    from photoheaven import __version__

    console.print(f"PhotoHeaven {__version__}")
