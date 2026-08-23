"""Face recognition subcommands for the PhotoHeaven CLI."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from photoheaven.adapters.face.insightface import InsightFaceAnalyzer
from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.face_clustering_service import FaceClusteringService
from photoheaven.application.face_detection_service import FaceDetectionService

logger = logging.getLogger(__name__)

faces_app = typer.Typer(
    name="faces",
    help="Detect, cluster, and name faces in your library.",
    no_args_is_help=True,
)
console = Console()


def _get_db_path(db_path: str | None) -> str:
    if db_path:
        resolved = db_path
    else:
        resolved = str(Path.cwd() / "db" / "photoheaven.db")
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    return resolved


@faces_app.command()
def detect(
    db_path: str | None = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-analyse media even if already processed."
    ),
    det_size: int = typer.Option(
        640, "--det-size", help="InsightFace detector input size (e.g. 320 or 640)."
    ),
    min_confidence: float = typer.Option(
        0.0,
        "--min-confidence",
        help="Minimum face detection confidence to keep (0.0 keeps all).",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        help="Number of media files to fetch per database query.",
    ),
) -> None:
    """Run face detection on unprocessed images in the library."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)

    try:
        analyzer = InsightFaceAnalyzer(det_size=(det_size, det_size))
    except RuntimeError as exc:
        console.print(f"[red]Failed to load face analysis model:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error initialising face analyser")
        console.print(
            "[red]Unexpected error loading face analysis model:[/red] "
            f"{exc}"
        )
        raise typer.Exit(1) from exc

    service = FaceDetectionService(analyzer=analyzer, repository=repository)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # The service reports progress internally via logging; the spinner
        # simply shows that work is happening.
        task = progress.add_task("Detecting faces...", total=None)
        result = service.detect(
            force=force,
            min_confidence=min_confidence,
            batch_size=batch_size,
        )
        progress.remove_task(task)

    table = Table(title="Face detection summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("Media processed", str(result.processed))
    table.add_row("Media skipped", str(result.skipped))
    table.add_row("Errors", str(result.errors))
    table.add_row("Faces detected", str(result.faces_detected))
    console.print(table)

    if result.errors:
        raise typer.Exit(1)


@faces_app.command()
def cluster(
    db_path: str | None = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
    eps: float = typer.Option(
        0.4,
        "--eps",
        help="Maximum cosine distance between two faces to be considered the same person.",
    ),
    min_samples: int = typer.Option(
        2,
        "--min-samples",
        help="Minimum faces required to form a cluster.",
    ),
    algorithm: str = typer.Option(
        "dbscan",
        "--algorithm",
        help="Clustering algorithm. Only dbscan is supported.",
    ),
) -> None:
    """Cluster detected faces into identity groups."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)
    service = FaceClusteringService(repository=repository)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Clustering faces...", total=None)
        try:
            result = service.cluster(eps=eps, min_samples=min_samples, algorithm=algorithm)
        except ValueError as exc:
            progress.remove_task(task)
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except Exception as exc:
            progress.remove_task(task)
            logger.exception("Face clustering failed")
            console.print(f"[red]Face clustering failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.remove_task(task)

    table = Table(title="Face clustering summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("Total faces", str(result.total_faces))
    table.add_row("Clusters", str(result.num_clusters))
    table.add_row("Clustered faces", str(result.clustered_faces))
    table.add_row("Noise / unclustered", str(result.noise_faces))
    console.print(table)
