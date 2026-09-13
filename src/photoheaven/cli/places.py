"""Place/scene identification subcommands for the PhotoHeaven CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import pathname2url

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.adapters.places import CompositePlaceIdentifier
from photoheaven.application.place_detection_service import PlaceDetectionService
from photoheaven.cli import config as cli_config
from photoheaven.domain.models import MediaFile, PlaceRecord

logger = logging.getLogger(__name__)

places_app = typer.Typer(
    name="places",
    help="Detect countries and scenes in your library.",
    no_args_is_help=True,
)
console = Console()


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


def _display_path(absolute_path: str) -> Text:
    """Return a short, clickable representation of *absolute_path*."""
    path = Path(absolute_path).expanduser().resolve()
    files_root = cli_config.resolve_library_files_root()

    try:
        if files_root is not None:
            label = str(path.relative_to(files_root))
        else:
            label = str(path)
    except ValueError:
        label = str(path)

    link = f"file://{pathname2url(str(path))}"
    return Text(label, style=f"cyan link {link}", no_wrap=True)


@places_app.command()
def detect(
    force: bool = typer.Option(
        False, "--force", help="Re-analyse media even if already processed."
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        help="Number of media files to fetch per database query (internal batching).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum total number of media files to consider.",
    ),
    random: bool = typer.Option(
        False,
        "--random",
        help="Pick media files at random (requires --limit).",
    ),
    scene_threshold: float = typer.Option(
        0.1,
        "--scene-threshold",
        help="Minimum Places365 confidence (0-1) before a scene label is stored.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print each analysed photo and its detected place.",
    ),
) -> None:
    """Run country/scene detection on unprocessed images in the library."""
    db = _get_db_path()
    repository = SqliteMediaRepository(db)

    if random and limit is None:
        console.print("[red]--random requires --limit.[/red]")
        raise typer.Exit(1)

    try:
        identifier = CompositePlaceIdentifier(scene_threshold=scene_threshold)
    except RuntimeError as exc:
        console.print(f"[red]Failed to load place identification model:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error initialising place identifier")
        console.print(
            "[red]Unexpected error loading place identification model:[/red] "
            f"{exc}"
        )
        raise typer.Exit(1) from exc

    service = PlaceDetectionService(identifier=identifier, repository=repository)

    detected_rows: list[tuple[str, str, str, str]] = []

    def _progress_callback(media: MediaFile, record: PlaceRecord) -> None:
        country = record.country or "—"
        scene = record.scene or "—"
        detected_rows.append((media.path, country, record.country_source or "—", scene))
        if verbose:
            console.print(
                f"[cyan]{Path(media.path).name}[/cyan] "
                f"→ [green]{country}[/green] / [magenta]{scene}[/magenta]"
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting places...", total=None)
        result = service.detect(
            force=force,
            batch_size=batch_size,
            limit=limit,
            random=random,
            progress_callback=_progress_callback,
        )
        progress.remove_task(task)

    table = Table(title="Place detection summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("Media processed", str(result.processed))
    table.add_row("Media skipped", str(result.skipped))
    table.add_row("Errors", str(result.errors))
    table.add_row("Countries identified", str(result.countries))
    table.add_row("Scenes identified", str(result.scenes))
    console.print(table)

    if detected_rows:
        results_table = Table(title=f"Detected places ({len(detected_rows)})")
        results_table.add_column("Path", style="cyan", no_wrap=True)
        results_table.add_column("Country", style="green")
        results_table.add_column("Source", style="yellow")
        results_table.add_column("Scene", style="magenta")
        for path, country, source, scene in detected_rows:
            results_table.add_row(
                _display_path(path), country, source, scene
            )
        console.print(results_table)

    if result.errors:
        raise typer.Exit(1)


@places_app.command()
def reset(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt."
    ),
) -> None:
    """Clear all place detection results and exit."""
    if not yes:
        typer.confirm(
            "Delete all place records and reset place-analysis flags?",
            abort=True,
        )

    db = _get_db_path()
    repository = SqliteMediaRepository(db)
    count = repository.reset_place_analysis()
    console.print(
        f"[yellow]Reset place detection for {count} media file(s).[/yellow]"
    )


@places_app.command("list")
def list_(
    limit: int = typer.Option(
        100, "--limit", "-l", help="Maximum number of rows to show."
    ),
    offset: int = typer.Option(
        0, "--offset", help="Skip this many rows."
    ),
) -> None:
    """List saved place records ordered by analysis time."""
    db = _get_db_path()
    repository = SqliteMediaRepository(db)

    rows = repository.list_place_records_summary(limit=limit, offset=offset)
    if not rows:
        console.print("[yellow]No place records found. Run 'ph places detect' first.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Place records (showing {len(rows)})")
    table.add_column("Path", style="cyan", no_wrap=True)
    table.add_column("Country", style="green")
    table.add_column("Source", style="yellow")
    table.add_column("Scene", style="magenta")

    for row in rows:
        table.add_row(
            _display_path(row["path"]),
            row["country"] or "—",
            row["country_source"] or "—",
            row["scene"] or "—",
        )

    console.print(table)
