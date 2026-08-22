"""Typer CLI entry point for PhotoHeaven."""

from __future__ import annotations

import logging
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
from photoheaven.domain.models import MediaFile, MediaType

app = typer.Typer(
    name="photoheaven",
    help="Organise, analyse, and face-cluster local photos and videos.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)
console = Console()

# Make library log messages visible through the CLI.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".3gp",
    ".webm",
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
) -> None:
    """Ingest photos and videos into the library."""
    service = _build_service(_get_db_path(db_path))
    files = _collect_files(path, recursive)

    if not files:
        console.print("[yellow]No supported media files found.[/yellow]")
        raise typer.Exit(0)

    counts = {"added": 0, "updated": 0, "skipped": 0, "error": 0}

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


@app.command()
def version() -> None:
    """Print the PhotoHeaven version."""
    from photoheaven import __version__

    console.print(f"PhotoHeaven {__version__}")
